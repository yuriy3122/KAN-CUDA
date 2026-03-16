#include "kan_model.h"
#include "kan_cuda_kernels.h"
#include <cuda_runtime.h>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

DeviceArray::DeviceArray(int n) { resize(n); }
DeviceArray::~DeviceArray() { if (ptr) cudaFree(ptr); }
DeviceArray::DeviceArray(DeviceArray&& o) noexcept { ptr=o.ptr; size=o.size; o.ptr=nullptr; o.size=0; }
DeviceArray& DeviceArray::operator=(DeviceArray&& o) noexcept { if (this!=&o){ if(ptr) cudaFree(ptr); ptr=o.ptr; size=o.size; o.ptr=nullptr; o.size=0;} return *this; }
void DeviceArray::resize(int n) { if (ptr) ck(cudaFree(ptr)); size=n; ck(cudaMalloc(&ptr, sizeof(float)*n)); }
void DeviceArray::from_host(const std::vector<float>& v){ if(size!=(int)v.size()) resize((int)v.size()); ck(cudaMemcpy(ptr,v.data(),sizeof(float)*v.size(),cudaMemcpyHostToDevice)); }
void DeviceArray::to_host(std::vector<float>& v) const { v.resize(size); ck(cudaMemcpy(v.data(),ptr,sizeof(float)*size,cudaMemcpyDeviceToHost)); }

DeviceIntArray::DeviceIntArray(int n){ resize(n);} DeviceIntArray::~DeviceIntArray(){ if(ptr) cudaFree(ptr);} 
void DeviceIntArray::resize(int n){ if(ptr) ck(cudaFree(ptr)); size=n; ck(cudaMalloc(&ptr,sizeof(int)*n)); }
void DeviceIntArray::from_host(const std::vector<int>& v){ if(size!=(int)v.size()) resize((int)v.size()); ck(cudaMemcpy(ptr,v.data(),sizeof(int)*v.size(),cudaMemcpyHostToDevice)); }

KANLayer::KANLayer(int in_dim_, int out_dim_, int grid_intervals_, int k_)
: in_dim(in_dim_), out_dim(out_dim_), grid_intervals(grid_intervals_), k(std::min(k_,3)) {
    n_knots = grid_intervals + 2 * k;
    n_coef = grid_intervals + k;
    grid.resize(in_dim * n_knots);
    coef.resize(in_dim * out_dim * n_coef);
    scale_base.resize(in_dim * out_dim);
    scale_sp.resize(in_dim * out_dim);
    grad_coef.resize(in_dim * out_dim * n_coef);
    grad_scale_base.resize(in_dim * out_dim);
    grad_scale_sp.resize(in_dim * out_dim);
}

void KANLayer::init(float low, float high, unsigned seed) {
    std::vector<float> hgrid(in_dim * n_knots), hcoef(in_dim * out_dim * n_coef), hsb(in_dim * out_dim), hss(in_dim * out_dim);
    for (int i=0;i<in_dim;++i) for(int t=0;t<n_knots;++t) hgrid[i*n_knots+t] = low + (high-low) * t / float(std::max(1,n_knots-1));
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.0f, 0.01f);
    for (auto& v : hcoef) v = nd(rng);
    std::fill(hsb.begin(), hsb.end(), 1.0f);
    std::fill(hss.begin(), hss.end(), 1.0f);
    grid.from_host(hgrid); coef.from_host(hcoef); scale_base.from_host(hsb); scale_sp.from_host(hss); zero_grads();
}

void KANLayer::forward(const DeviceArray& x, DeviceArray& out, int batch) const {
    if (out.size != batch * out_dim) out.resize(batch * out_dim);
    kan_forward_cuda(x.ptr, grid.ptr, coef.ptr, scale_base.ptr, scale_sp.ptr, out.ptr, batch, in_dim, out_dim, n_knots, n_coef, k);
}

void KANLayer::backward(const DeviceArray& grad_out, const DeviceArray& x, DeviceArray& grad_x, int batch) {
    if (grad_x.size != batch * in_dim) grad_x.resize(batch * in_dim);
    zero_cuda(grad_x.ptr, grad_x.size);
    kan_backward_cuda(grad_out.ptr, x.ptr, grid.ptr, coef.ptr, scale_base.ptr, scale_sp.ptr,
                      grad_x.ptr, grad_coef.ptr, grad_scale_base.ptr, grad_scale_sp.ptr,
                      batch, in_dim, out_dim, n_knots, n_coef, k);
}

void KANLayer::step(float lr, float weight_decay) {
    sgd_update_cuda(coef.ptr, grad_coef.ptr, grad_coef.size, lr, weight_decay);
    sgd_update_cuda(scale_base.ptr, grad_scale_base.ptr, grad_scale_base.size, lr, weight_decay);
    sgd_update_cuda(scale_sp.ptr, grad_scale_sp.ptr, grad_scale_sp.size, lr, weight_decay);
}
void KANLayer::zero_grads() {
    zero_cuda(grad_coef.ptr, grad_coef.size);
    zero_cuda(grad_scale_base.ptr, grad_scale_base.size);
    zero_cuda(grad_scale_sp.ptr, grad_scale_sp.size);
}

KANClassifier::KANClassifier(int input_dim_, int hidden_dim_, int grid_, int k_)
: l1(input_dim_, hidden_dim_, grid_, k_), l2(hidden_dim_, num_classes, grid_, k_), input_dim(input_dim_), hidden_dim(hidden_dim_) {
    l1.init(-1.0f, 1.0f, 42u); l2.init(-1.0f, 1.0f, 123u);
}

float KANClassifier::train_epoch(const std::vector<float>& X, const std::vector<int>& y, int rows, int cols, int batch_size, float lr) {
    std::vector<int> order(rows); std::iota(order.begin(), order.end(), 0); std::mt19937 rng(5u); std::shuffle(order.begin(), order.end(), rng);
    std::vector<float> batchX; std::vector<int> batchY;
    DeviceArray dx, h1, logits, grad_logits, grad_h1; DeviceIntArray dy; DeviceArray dloss(1);
    float total_loss = 0.0f;
    for (int st = 0; st < rows; st += batch_size) {
        int B = std::min(batch_size, rows - st);
        batchX.resize(B * cols); batchY.resize(B);
        for (int i = 0; i < B; ++i) {
            int src = order[st + i];
            std::copy_n(&X[src * cols], cols, &batchX[i * cols]);
            batchY[i] = y[src];
        }
        dx.from_host(batchX); dy.from_host(batchY);
        l1.forward(dx, h1, B);
        l2.forward(h1, logits, B);
        if (grad_logits.size != logits.size) grad_logits.resize(logits.size);
        softmax_xent_grad_cuda(logits.ptr, dy.ptr, grad_logits.ptr, dloss.ptr, B, num_classes);
        std::vector<float> hloss; dloss.to_host(hloss); total_loss += hloss[0];
        l1.zero_grads(); l2.zero_grads();
        l2.backward(grad_logits, h1, grad_h1, B);
        l1.backward(grad_h1, dx, dx, B);
        l2.step(lr); l1.step(lr);
        ck(cudaDeviceSynchronize());
    }
    return total_loss / rows;
}

float KANClassifier::evaluate_accuracy(const std::vector<float>& X, const std::vector<int>& y, int rows, int cols, int batch_size) const {
    DeviceArray dx, h1, logits;
    int correct = 0;
    std::vector<float> batchX, hlogits;
    for (int st = 0; st < rows; st += batch_size) {
        int B = std::min(batch_size, rows - st);
        batchX.assign(X.begin() + st * cols, X.begin() + (st + B) * cols);
        dx.from_host(batchX);
        l1.forward(dx, h1, B);
        l2.forward(h1, logits, B);
        logits.to_host(hlogits);
        for (int i = 0; i < B; ++i) {
            int pred = hlogits[i * num_classes + 1] > hlogits[i * num_classes] ? 1 : 0;
            correct += (pred == y[st + i]);
        }
    }
    return rows ? float(correct) / rows : 0.0f;
}
