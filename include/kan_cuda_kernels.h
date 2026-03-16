#pragma once
#include <cuda_runtime.h>

void kan_forward_cuda(
    const float* x,
    const float* grid,
    const float* coef,
    const float* scale_base,
    const float* scale_sp,
    float* out,
    int B, int I, int O,
    int GKNOT, int GCOEF, int K);

void kan_backward_cuda(
    const float* grad_out,
    const float* x,
    const float* grid,
    const float* coef,
    const float* scale_base,
    const float* scale_sp,
    float* grad_x,
    float* grad_coef,
    float* grad_scale_base,
    float* grad_scale_sp,
    int B, int I, int O,
    int GKNOT, int GCOEF, int K);

void sgd_update_cuda(float* param, const float* grad, int n, float lr, float weight_decay);
void zero_cuda(float* ptr, int n);
void softmax_xent_grad_cuda(const float* logits, const int* labels, float* grad_logits, float* loss, int B, int C);
