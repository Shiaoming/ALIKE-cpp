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
#pragma once

#include <math.h>
#include <torch/torch.h>

namespace alike
{
    class SimplePadder
    {
    public:
        SimplePadder(int h, int w, int divis_by = 32)
        {
            mH = (h % divis_by == 0) ? h : (h / divis_by + 1) * divis_by;
            mW = (w % divis_by == 0) ? w : (w / divis_by + 1) * divis_by;
        }

        torch::Tensor pad(torch::Tensor &x)
        {
            // x: BCHW
            auto device = x.device();
            int64_t b = x.size(0);
            int64_t c = x.size(1);
            int64_t h = x.size(2);
            int64_t w = x.size(3);
            if (mH != h)
            {
                auto h_padding = torch::zeros({b, c, mH - h, w}).to(device);
                x = torch::cat({x, h_padding}, 2);
            }
            if (mW != w)
            {
                auto w_padding = torch::zeros({b, c, mH, mW - w}).to(device);
                x = torch::cat({x, w_padding}, 3);
            }
            return x;
        }

        torch::Tensor unpad(torch::Tensor &x)
        {
            using namespace torch::indexing;

            int64_t h = x.size(2);
            int64_t w = x.size(3);
            if (mH != h or mW != w)
                x = x.index({Slice(), Slice(), Slice(None, h), Slice(None, w)}); // [:, :, :h, :w]
            return x;
        }

    private:
        int mH,
            mW;
    };
}