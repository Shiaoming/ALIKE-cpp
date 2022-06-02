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
#include <omp.h>
#include <torch/torch.h>
#include <get_patches.h>

// map: CxHxW
// points: Nx2
// radius: int
// return: NxCx(2*radius+1)x(2*radius+1)
// torch implementation: too slow!!
torch::Tensor get_patches_torch(const torch::Tensor &map,
                                torch::Tensor &points,
                                int radius)
{
    namespace F = torch::nn::functional;
    using namespace torch::indexing;

    auto N = points.size(0);
    auto C = map.size(0);
    auto kernel_size = 2 * radius + 1;

    // pad map
    auto options = F::PadFuncOptions({radius, radius, radius, radius}).mode(torch::kReflect);
    auto map_pad = F::pad(map.unsqueeze(0), options).squeeze(0); // Cx(H+2*radius)x(W+2*radius)

    // get patches
    torch::Tensor patches = torch::zeros({N, C, kernel_size, kernel_size}, map.options());
    // float *p_points = points.accessor<float, 2>().data();   // Nx2
    // float *p_map = map_pad.accessor<float, 3>().data();     // Cx(H+2*radius)x(W+2*radius)
    // float *p_patches = patches.accessor<float, 4>().data(); // NxCx(2*radius+1)x(2*radius+1)
    // too slow!!!
    for (auto i = 0; i < N; i++)
    {
        auto w_start = points.index({i, 0}).item().toLong();
        auto h_start = points.index({i, 1}).item().toLong();
        torch::Tensor patch = map_pad.index({Slice(),
                                             Slice(h_start, h_start + kernel_size),
                                             Slice(w_start, w_start + kernel_size)});
        // std::cout << patch.sizes() << std::endl;
        // std::cout << patch << std::endl;
        patches.index_put_({i}, patch);
    }
    return patches;
}

// map: CxHxW
// points: Nx2
// radius: int
// return: NxCx(2*radius+1)x(2*radius+1)
torch::Tensor get_patches_cpu(const torch::Tensor &map,
                              torch::Tensor &points,
                              int radius)
{
    namespace F = torch::nn::functional;
    using namespace torch::indexing;

    auto device = map.device();
    auto N = points.size(0);
    auto C = map.size(0);
    auto H = map.size(1);
    auto W = map.size(2);
    auto kernel_size = 2 * radius + 1;

    // to cpu:
    torch::Tensor map_cpu = map.to(torch::kCPU);
    torch::Tensor points_cpu = points.to(torch::kCPU);

    // pad map
    auto options = F::PadFuncOptions({radius, radius, radius, radius}).mode(torch::kReflect);
    auto map_pad = F::pad(map_cpu.unsqueeze(0), options).squeeze(0); // Cx(H+2*radius)x(W+2*radius)

    // get patches
    torch::Tensor patches = torch::zeros({N, C, kernel_size, kernel_size}, map_cpu.options());
    long *p_points = points_cpu.accessor<long, 2>().data(); // Nx2
    float *p_map = map_pad.accessor<float, 3>().data();     // Cx(H+2*radius)x(W+2*radius)
    float *p_patches = patches.accessor<float, 4>().data(); // NxCxkernel_sizexkernel_size

#pragma omp parallel for
    for (auto in = 0; in < N; in++)
    {
        long w_start = static_cast<long>(*(p_points + in * 2 + 0));
        long h_start = static_cast<long>(*(p_points + in * 2 + 1));

        // copy data
        for (auto ic = 0; ic < C; ic++)
        {
            for (auto ih = 0; ih < kernel_size; ih++)
            {
                for (auto iw = 0; iw < kernel_size; iw++)
                {
                    // Cx(H+2*radius)x(W+2*radius)
                    float *p_src = p_map +
                                   ic * (H + 2 * radius) * (W + 2 * radius) +
                                   (ih + h_start) * (W + 2 * radius) +
                                   (iw + w_start);
                    // NxCxkernel_sizexkernel_size
                    float *p_dst = p_patches +
                                   in * C * kernel_size * kernel_size +
                                   ic * kernel_size * kernel_size +
                                   ih * kernel_size +
                                   iw;
                    memcpy(p_dst, p_src, sizeof(float));
                }
            }
        }
    }
    return patches.to(device);
}

torch::Tensor get_patches(const torch::Tensor &map,
                          torch::Tensor &points,
                          int radius)
{
    if (map.device() == torch::kCPU)
        return get_patches_cpu(map, points, radius);
    else
        return get_patches_cuda(map, points, radius);
}