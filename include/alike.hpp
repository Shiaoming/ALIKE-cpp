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

#include <string>

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <simple_padder.hpp>
#include <soft_detect.hpp>
#include <utils.h>

namespace alike
{
    class ALIKE
    {
    public:
        ALIKE(std::string model_path,
              bool cuda = false,
              int radius = 2,
              int top_k = 500,
              float scores_th = 0.1,
              int n_limit = 9999,
              bool subpixel = false) : mJitModel(torch::jit::load(model_path)),
                                       mDevice((cuda) ? torch::kCUDA : torch::kCPU),
                                       mSubPixel(subpixel),
                                       mDkd(radius, top_k, scores_th, n_limit, subpixel)
        {
            mJitModel.to(mDevice);
            mJitModel.eval();
        }

        void extract(torch::Tensor &img_tensor,
                     torch::Tensor &score_map,
                     torch::Tensor &descriptor_map)
        {
            torch::NoGradGuard nograd; // equivalent to "with torch.nograd()" in python

            auto padder = SimplePadder(img_tensor.size(2), img_tensor.size(3), 32);
            img_tensor = padder.pad(img_tensor);

            auto result = mJitModel.forward({img_tensor}).toTensor(); // 1x(dim+1)xHxW

            using namespace torch::indexing;
            score_map = result.index({Slice(), Slice(-1, None), Slice(), Slice()});      // result[0,-1,:,:]
            descriptor_map = result.index({Slice(), Slice(None, -1), Slice(), Slice()}); // result[0,:-1,:,:]

            score_map = padder.unpad(score_map);
            descriptor_map = padder.unpad(descriptor_map);
        }

        void detect(torch::Tensor &score_map,
                    torch::Tensor &keypoints,
                    torch::Tensor &dispersitys,
                    torch::Tensor &kptscores)
        {
            mDkd.detect_keypoints(score_map, keypoints, dispersitys, kptscores);

            // sort
            using namespace torch::indexing;
            auto indices = torch::argsort(kptscores, -1, true);
            keypoints = keypoints.index({indices});
            dispersitys = dispersitys.index({indices});
            kptscores = kptscores.index({indices});
        }

        void compute(torch::Tensor &descriptor_map,
                     torch::Tensor &keypoints,
                     torch::Tensor &descriptors)
        {
            using namespace torch::indexing;
            namespace F = torch::nn::functional;

            auto device = descriptor_map.device();
            int64_t b = descriptor_map.size(0); // b=1
            int64_t h = descriptor_map.size(2);
            int64_t w = descriptor_map.size(3);
            int64_t wh_int[] = {w - 1, h - 1};
            auto wh = torch::from_blob(wh_int, {2}, torch::kLong).to(device);

            auto kpts = keypoints;
            torch::Tensor descs;
            if (!mSubPixel)
            {
                kpts = (kpts + 1) / 2 * wh;
                auto x = kpts.index({Slice(), 1});
                auto y = kpts.index({Slice(), 0});
                descs = descriptor_map.index({0, Slice(), x.to(torch::kLong), y.to(torch::kLong)});
            }
            else
            {
                auto options = F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true);
                descs = F::grid_sample(descriptor_map.index({0}).unsqueeze(0),
                                       kpts.view({1, 1, -1, 2}),
                                       options)
                            .index({0, Slice(), 0, Slice()});
            }
            descs = F::normalize(descs, F::NormalizeFuncOptions().p(2).dim(0));
            descriptors = descs.t();
        }

        void detectAndCompute(torch::Tensor &score_map,
                              torch::Tensor &descriptor_map,
                              torch::Tensor &keypoints,
                              torch::Tensor &dispersitys,
                              torch::Tensor &kptscores,
                              torch::Tensor &descriptors)
        {
            auto device = descriptor_map.device();
            int64_t b = descriptor_map.size(0); // b=1
            int64_t h = descriptor_map.size(2);
            int64_t w = descriptor_map.size(3);
            int64_t wh_int[] = {w - 1, h - 1};
            auto wh = torch::from_blob(wh_int, {2}, torch::kLong).to(device);

            detect(score_map, keypoints, dispersitys, kptscores);
            compute(descriptor_map, keypoints, descriptors);

            // -1~1 -> wxh
            keypoints = (keypoints + 1) / 2 * wh;
        }

        void extactAndDetectAndCompute(torch::Tensor &img_tensor,
                                       torch::Tensor &keypoints,
                                       torch::Tensor &dispersitys,
                                       torch::Tensor &kptscores,
                                       torch::Tensor &descriptors)
        {
            torch::Tensor score_map, descriptor_map;

            extract(img_tensor, score_map, descriptor_map);
            detectAndCompute(score_map, descriptor_map, keypoints, dispersitys, kptscores, descriptors);
        }

        void toOpenCVFormat(torch::Tensor &keypoints_t,
                            torch::Tensor &dispersitys_t,
                            torch::Tensor &kptscores_t,
                            torch::Tensor &descriptors_t,
                            std::vector<cv::KeyPoint> &keypoints, // x,y; size: dispersity; response: score
                            cv::Mat &descriptors)
        {
            // to cpu
            keypoints_t = keypoints_t.to(torch::kCPU);
            dispersitys_t = dispersitys_t.to(torch::kCPU);
            kptscores_t = kptscores_t.to(torch::kCPU);

            auto num = keypoints_t.size(0);
            auto keypoints_a = keypoints_t.accessor<float, 2>();
            auto bdispersitys_a = dispersitys_t.accessor<float, 1>();
            auto kptscores_a = kptscores_t.accessor<float, 1>();
            for (auto i = 0; i < num; i++)
            {
                cv::KeyPoint keypoint = cv::KeyPoint(keypoints_a[i][0], keypoints_a[i][1], bdispersitys_a[i], -1, kptscores_a[i]);
                keypoints.push_back(keypoint);
            }
            descriptors = tensor2Mat(descriptors_t);
        }

    private:
        DKD mDkd;
        bool mSubPixel;
        torch::jit::script::Module mJitModel;
        torch::Device mDevice;
    };
}