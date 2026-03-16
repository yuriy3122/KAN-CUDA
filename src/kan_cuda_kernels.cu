#include "kan_cuda_kernels.h"
#include <cuda_runtime.h>
#include <cmath>

namespace {
constexpr int B_TILE = 16;
constexpr int O_TILE = 8;
constexpr int MAX_K = 3;
constexpr int MAX_BASIS = 64;

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float dsilu(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s + x * s * (1.0f - s);
}
__device__ int find_span(const float* knots, int n_knots, int k, float x) {
    int low = k, high = n_knots - k - 1;
    if (high <= low) return low;
    if (x <= knots[low]) return low;
    if (x >= knots[high - 1]) return high - 1;
    int left = low, right = high - 1;
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (x < knots[mid]) right = mid - 1;
        else if (x >= knots[mid + 1]) left = mid + 1;
        else return mid;
    }
    return low;
}
__device__ void bspline_local_basis(float x, const float* knots, int, int k, int span, float* basis) {
    float left[MAX_K + 1], right[MAX_K + 1];
    basis[0] = 1.0f;
    #pragma unroll
    for (int j = 1; j <= MAX_K; ++j) {
        if (j > k) break;
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        float saved = 0.0f;
        #pragma unroll
        for (int r = 0; r < j; ++r) {
            float denom = right[r + 1] + left[j - r];
            float temp = fabsf(denom) < 1e-12f ? 0.0f : basis[r] / denom;
            basis[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        basis[j] = saved;
    }
}
__device__ float spline_eval_derivative_single(float x, const float* knots, const float* coef_vec, int GKNOT, int GCOEF, int K) {
    if (K <= 0) return 0.0f;
    int span = find_span(knots, GKNOT, K, x);
    float basis_km1[MAX_K + 1] = {0};
    bspline_local_basis(x, knots, GKNOT, K - 1, span, basis_km1);
    int first_coef = span - K;
    float deriv = 0.0f;
    #pragma unroll
    for (int j = 0; j <= MAX_K; ++j) {
        if (j > K) break;
        int g = first_coef + j;
        float dBj = 0.0f;
        if (j >= 1) {
            float denom1 = knots[g + K] - knots[g];
            if (fabsf(denom1) >= 1e-12f) dBj += (K / denom1) * basis_km1[j - 1];
        }
        if (j <= K - 1) {
            float denom2 = knots[g + K + 1] - knots[g + 1];
            if (fabsf(denom2) >= 1e-12f) dBj -= (K / denom2) * basis_km1[j];
        }
        if (g >= 0 && g < GCOEF) deriv += coef_vec[g] * dBj;
    }
    return deriv;
}

__global__ void zero_kernel(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}
__global__ void sgd_kernel(float* p, const float* g, int n, float lr, float wd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] -= lr * (g[i] + wd * p[i]);
}
__global__ void softmax_xent_kernel(const float* logits, const int* labels, float* grad, float* loss, int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float* row = logits + b * C;
    float m = row[0];
    for (int c = 1; c < C; ++c) m = fmaxf(m, row[c]);
    float s = 0.0f;
    for (int c = 0; c < C; ++c) s += expf(row[c] - m);
    float logsum = m + logf(s);
    atomicAdd(loss, -(row[labels[b]] - logsum));
    for (int c = 0; c < C; ++c) {
        float p = expf(row[c] - logsum);
        grad[b * C + c] = (p - (c == labels[b] ? 1.0f : 0.0f)) / B;
    }
}

__global__ void kan_forward_kernel(const float* x,const float* grid,const float* coef,const float* scale_base,const float* scale_sp,float* out,
    int B,int I,int O,int GKNOT,int GCOEF,int K) {
    const int b_local = threadIdx.x, o_local = threadIdx.y;
    const int b = blockIdx.x * B_TILE + b_local;
    const int o = blockIdx.y * O_TILE + o_local;
    extern __shared__ float smem[];
    float* s_grid = smem;
    float* s_coef = s_grid + GKNOT;
    float acc = 0.0f;
    if (b < B && o < O) {
        for (int i = 0; i < I; ++i) {
            for (int t = b_local + o_local * B_TILE; t < GKNOT; t += B_TILE * O_TILE) s_grid[t] = grid[i * GKNOT + t];
            int o_base = blockIdx.y * O_TILE;
            for (int t = b_local + o_local * B_TILE; t < O_TILE * GCOEF; t += B_TILE * O_TILE) {
                int oo = t / GCOEF, cc = t % GCOEF, go = o_base + oo;
                s_coef[t] = go < O ? coef[(i * O + go) * GCOEF + cc] : 0.0f;
            }
            __syncthreads();
            float xv = x[b * I + i], base = silu(xv);
            int span = find_span(s_grid, GKNOT, K, xv);
            float basis[MAX_K + 1] = {0};
            bspline_local_basis(xv, s_grid, GKNOT, K, span, basis);
            float spline = 0.0f;
            int first_coef = span - K;
            #pragma unroll
            for (int j = 0; j <= MAX_K; ++j) {
                if (j > K) break;
                int cidx = first_coef + j;
                if (cidx >= 0 && cidx < GCOEF) spline += s_coef[o_local * GCOEF + cidx] * basis[j];
            }
            acc += scale_base[i * O + o] * base + scale_sp[i * O + o] * spline;
            __syncthreads();
        }
        out[b * O + o] = acc;
    }
}

__global__ void kan_backward_kernel(const float* grad_out,const float* x,const float* grid,const float* coef,const float* scale_base,const float* scale_sp,
    float* grad_x,float* grad_coef,float* grad_scale_base,float* grad_scale_sp,
    int B,int I,int O,int GKNOT,int GCOEF,int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * O;
    if (idx >= total) return;
    int b = idx / O, o = idx % O;
    float g = grad_out[b * O + o];
    if (g == 0.0f) return;
    for (int i = 0; i < I; ++i) {
        const float* knots = grid + i * GKNOT;
        const float* coef_vec = coef + (i * O + o) * GCOEF;
        float xv = x[b * I + i], sb = scale_base[i * O + o], ss = scale_sp[i * O + o];
        float base = silu(xv), dbase = dsilu(xv);
        int span = find_span(knots, GKNOT, K, xv);
        float basis[MAX_K + 1] = {0};
        bspline_local_basis(xv, knots, GKNOT, K, span, basis);
        float spline = 0.0f;
        int first_coef = span - K;
        #pragma unroll
        for (int j = 0; j <= MAX_K; ++j) {
            if (j > K) break;
            int cidx = first_coef + j;
            if (cidx >= 0 && cidx < GCOEF) {
                float bj = basis[j];
                float c = coef_vec[cidx];
                spline += c * bj;
                atomicAdd(&grad_coef[(i * O + o) * GCOEF + cidx], g * ss * bj);
            }
        }
        atomicAdd(&grad_scale_base[i * O + o], g * base);
        atomicAdd(&grad_scale_sp[i * O + o], g * spline);
        float dspline = spline_eval_derivative_single(xv, knots, coef_vec, GKNOT, GCOEF, K);
        atomicAdd(&grad_x[b * I + i], g * (sb * dbase + ss * dspline));
    }
}
}

void kan_forward_cuda(const float* x,const float* grid,const float* coef,const float* scale_base,const float* scale_sp,float* out,
    int B,int I,int O,int GKNOT,int GCOEF,int K) {
    dim3 block(B_TILE, O_TILE);
    dim3 grid_dim((B + B_TILE - 1) / B_TILE, (O + O_TILE - 1) / O_TILE);
    size_t shmem = (GKNOT + O_TILE * GCOEF) * sizeof(float);
    kan_forward_kernel<<<grid_dim, block, shmem>>>(x, grid, coef, scale_base, scale_sp, out, B, I, O, GKNOT, GCOEF, K);
}
void kan_backward_cuda(const float* grad_out,const float* x,const float* grid,const float* coef,const float* scale_base,const float* scale_sp,
    float* grad_x,float* grad_coef,float* grad_scale_base,float* grad_scale_sp,
    int B,int I,int O,int GKNOT,int GCOEF,int K) {
    int total = B * O, threads = 256, blocks = (total + threads - 1) / threads;
    kan_backward_kernel<<<blocks, threads>>>(grad_out, x, grid, coef, scale_base, scale_sp, grad_x, grad_coef, grad_scale_base, grad_scale_sp, B, I, O, GKNOT, GCOEF, K);
}
void sgd_update_cuda(float* param, const float* grad, int n, float lr, float weight_decay) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    sgd_kernel<<<blocks, threads>>>(param, grad, n, lr, weight_decay);
}
void zero_cuda(float* ptr, int n) { int threads = 256, blocks = (n + threads - 1) / threads; zero_kernel<<<blocks, threads>>>(ptr, n); }
void softmax_xent_grad_cuda(const float* logits, const int* labels, float* grad_logits, float* loss, int B, int C) {
    cudaMemset(loss, 0, sizeof(float));
    int threads = 256, blocks = (B + threads - 1) / threads;
    softmax_xent_kernel<<<blocks, threads>>>(logits, labels, grad_logits, loss, B, C);
}
