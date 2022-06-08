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

#include <get_patches.h>
#include <torch/torch.h>

namespace alike
{
    class DKD
    {
    public:
        // Differentiable keypoint detect module.
        // Args:
        // radius: soft detection radius, kernel size is (2 * radius + 1)
        // top_k: top_k > 0: return top k keypoints
        // scores_th: top_k <= 0 threshold mode:  scores_th > 0: return keypoints with scores>scores_th
        //                                        else: return keypoints with scores > scores.mean()
        // n_limit: max number of keypoint in threshold mode
        DKD(int radius,
            int top_k,
            float scores_th,
            int n_limit,
            bool subpixel = false) : mRadius(radius),
                                     mTopK(top_k), mSth(scores_th),
                                     mNLimit(n_limit),
                                     mSubPixel(subpixel)
        {
            mKernelSize = 2 * mRadius + 1;
            auto tmp = torch::linspace(-mRadius, mRadius, mKernelSize);
            mHWGrid = torch::stack(torch::meshgrid({tmp, tmp}, "ij")).view({2, -1}).t();

            using namespace torch::indexing;
            mHWGrid = mHWGrid.index({Slice(), torch::tensor({1, 0})}); //[:, [1, 0]]
        }

        void detect_keypoints(torch::Tensor &score_map,
                              torch::Tensor &keypoints,
                              torch::Tensor &dispersitys,
                              torch::Tensor &kptscores)
        {
            auto device = score_map.device();
            int64_t b = score_map.size(0); //  b=1
            int64_t h = score_map.size(2);
            int64_t w = score_map.size(3);
            int64_t wh_int[] = {w - 1, h - 1};
            auto wh = torch::from_blob(wh_int, {2}, torch::kLong).to(device);

            auto nms_scores = simpleNMS(score_map);

            // remove border
            using namespace torch::indexing;
            nms_scores.index_put_({Slice(), Slice(), Slice(None, mRadius + 1), Slice()}, 0); //[:, :, :self.radius + 1, :]
            nms_scores.index_put_({Slice(), Slice(), Slice(), Slice(None, mRadius + 1)}, 0); //[:, :, :, :self.radius + 1]
            nms_scores.index_put_({Slice(), Slice(), Slice(h - mRadius, None), Slice()}, 0); //[:, :, h - self.radius:, :]
            nms_scores.index_put_({Slice(), Slice(), Slice(), Slice(w - mRadius, None)}, 0); //[:, :, :, w - self.radius:]

            std::vector<torch::Tensor> indices_keypoints;
            // detect keypoints without grad
            if (mTopK > 0)
            {
                auto topk = torch::topk(nms_scores.view({b, -1}), mTopK);
                auto bindices = std::get<1>(topk); // b x mTopK
                for (auto i = 0; i < b; i++)
                {
                    indices_keypoints.push_back(bindices.index({i, Slice()}));
                }
            }
            else
            {
                torch::Tensor masks = nms_scores > mSth;
                masks = masks.reshape({b, -1});

                auto scores_view = score_map.reshape({b, -1});
                for (auto i = 0; i < b; i++)
                {
                    auto mask = masks.index({i});
                    auto scores = scores_view.index({i});
                    auto indices = torch::nonzero(mask).index({Slice(), 0});
                    if (indices.size(0) > mNLimit)
                    {
                        auto kpt_sc = scores.index({indices});
                        // kpts_sc.sort(descending=True)[1]
                        auto sort_idx = std::get<1>(torch::sort(kpt_sc, -1, true));
                        // sel_idx = sort_idx[:self.n_limit]
                        auto sel_idx = sort_idx.index({Slice(None, mNLimit)});
                        indices = indices.index({sel_idx});
                    }
                    indices_keypoints.push_back(indices);
                }
            }

            for (auto i = 0; i < b; i++)
            {
                namespace F = torch::nn::functional;

                torch::Tensor keypoints_wh;
                auto indices_kpt = indices_keypoints[i];
                auto keypoints_wh_nms = torch::stack({indices_kpt % w, torch::div(indices_kpt, w, "floor")}, 1);

                if (mSubPixel)
                {
                    mHWGrid = mHWGrid.to(device);
                    auto N = keypoints_wh_nms.size(0);
                    auto C = score_map.size(0); // C=1
                    auto kernel_size = 2 * mRadius + 1;

                    // NxCx(2*radius+1)x(2*radius+1)
                    auto patches = get_patches(score_map.index({i}), keypoints_wh_nms, mRadius);
                    auto patches_col = patches.reshape({N, -1});        // Nx(kernel_size**2)
                    auto max_vi = patches_col.max(-1, true);            // values, indices
                    auto max_v = std::get<0>(max_vi);                   // Nx1
                    auto x_exp = ((patches_col - max_v) / mTemp).exp(); // Nx(kernel_size**2)

                    // \frac{ \sum{(i,j) \times \exp(x/T)} }{ \sum{\exp(x/T)} }
                    auto wh_residual = x_exp.matmul(mHWGrid) / x_exp.sum(1, true); // Nx2, soft-argmax

                    // compute keypoints
                    keypoints_wh = keypoints_wh_nms + wh_residual;

                    // compute dispersitys
                    auto hw_grid_dist2 = torch::norm((mHWGrid.unsqueeze(0) - wh_residual.unsqueeze(1)) / mRadius,
                                                     2, -1);
                    hw_grid_dist2 = torch::square_(hw_grid_dist2);
                    dispersitys = (x_exp * hw_grid_dist2).sum(1) / x_exp.sum(1);
                }
                else
                {
                    keypoints_wh = keypoints_wh_nms;
                }

                // get normalized keypoints
                auto keypoints_xy = keypoints_wh / wh * 2 - 1; // (w, h)->(-1 ~1, -1 ~1)
                keypoints = keypoints_xy;

                // get score of keypoints
                auto options = F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true);
                auto kptscore = F::grid_sample(score_map.index({i}).unsqueeze(0),
                                               keypoints_xy.view({1, 1, -1, 2}),
                                               options); // CxN
                kptscores = kptscore.index({0, 0, 0, Slice()});

                // set dispersitys to zero for no subpixel mode
                if (!mSubPixel)
                    dispersitys = torch::zeros_like(kptscores);
            }
        }

        torch::Tensor simpleNMS(torch::Tensor &score_map)
        {
            namespace F = torch::nn::functional;

            auto zeros = torch::zeros_like(score_map);
            auto options = F::MaxPool2dFuncOptions(mRadius * 2 + 1).stride(1).padding(mRadius);
            auto scores_max_pool = F::max_pool2d(score_map, options);
            auto max_mask = (score_map == scores_max_pool);

            return torch::where(max_mask, score_map, zeros);
        }

    private:
        int mRadius;
        int mKernelSize;
        int mTopK;
        float mSth;
        int mNLimit;
        float mTemp = 0.1;
        torch::Tensor mHWGrid;
        bool mSubPixel;
    };
}