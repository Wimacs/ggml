#pragma once
#include "common.cuh"

#define CUDA_PIXEL_SHUFFLE_3D_BLOCK_SIZE 256
void ggml_cuda_op_pixel_shuffle_3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
