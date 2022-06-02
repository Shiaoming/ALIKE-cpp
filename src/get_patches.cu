/*
https://github.com/Shiaoming/ALIKE-cpp
BSD 3-Clause License

Copyright (c) 2022, Zhao Xiaoming
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <stdio.h>

namespace F = torch::nn::functional;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

const int blockSize = 256;

template <typename scalar_t>
__global__ void get_patches_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> map_pad, // Cx(H+2*radius)x(W+2*radius)
    const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> points,   // Nx2
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> patches,       // NxCxkernel_sizexkernel_size
    int radius)
{
    const int in = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = points.size(0);
    const int C = map_pad.size(0);
    const int kernel_size = 2 * radius + 1;

    if (in < N)
    {
        long w_start = points[in][0];
        long h_start = points[in][1];

        // copy data
        for (long ic = 0; ic < C; ic++)
        {
            for (long ih = 0; ih < kernel_size; ih++)
            {
                for (long iw = 0; iw < kernel_size; iw++)
                {
                    patches[in][ic][ih][iw] = map_pad[ic][h_start + ih][w_start + iw];
                }
            }
        }
    }
}

torch::Tensor get_patches_cuda(const torch::Tensor &map,
                               torch::Tensor &points,
                               int radius)
{
    CHECK_INPUT(map);
    CHECK_INPUT(points);

    auto N = points.size(0);
    auto C = map.size(0);
    auto kernel_size = 2 * radius + 1;

    // pad map
    auto options = F::PadFuncOptions({radius, radius, radius, radius}).mode(torch::kReflect);
    auto map_pad = F::pad(map.unsqueeze(0), options).squeeze(0); // Cx(H+2*radius)x(W+2*radius)

    // create patches
    torch::Tensor patches = torch::zeros({N, C, kernel_size, kernel_size}, map.options());
    const int threads = blockSize;
    const int blocks = (N + threads - 1) / threads;

    // cuda kernel
    AT_DISPATCH_FLOATING_TYPES(map_pad.type(),
                               "get_patches_cuda",
                               ([&]
                                { get_patches_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                      map_pad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                      points.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                                      patches.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                                      radius); }));

    // get error
    cudaDeviceSynchronize();
    cudaError_t cudaerr = cudaGetLastError();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    return patches;
}