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

#include <opencv2/opencv.hpp>

namespace alike
{
    class SimpleTracker
    {
    public:
        SimpleTracker(float match_th = 0.7) : mMth(match_th) {}

        int update(cv::Mat &img,
                   std::vector<cv::KeyPoint> &pts,
                   cv::Mat &desc)
        {
            auto N_matches = 0;
            if (cnt == 0)
            {
                for (auto i = 0; i < pts.size(); i++)
                {
                    auto pt = pts[i].pt;
                    cv::circle(img, pt, 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
                }
            }
            else
            {
                cv::FlannBasedMatcher matcher;
                std::vector<std::vector<cv::DMatch>> knn_matches;
                matcher.knnMatch(desc_prev, desc, knn_matches, 2);

                std::vector<cv::DMatch> good_matches;
                for (auto i = 0; i < knn_matches.size(); i++)
                {
                    if (knn_matches[i][0].distance < mMth * knn_matches[i][1].distance)
                    {
                        good_matches.push_back(knn_matches[i][0]);
                    }
                }

                N_matches = good_matches.size();
                for (auto i = 0; i < N_matches; i++)
                {
                    auto match = good_matches[i];
                    auto p1 = pts_prev[match.queryIdx].pt;
                    auto p2 = pts[match.trainIdx].pt;

                    cv::line(img, p1, p2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                    cv::circle(img, p2, 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
                }
            }

            pts_prev = pts;
            desc_prev = desc.clone();
            cnt++;
            return N_matches;
        }

    private:
        int cnt = 0;
        std::vector<cv::KeyPoint> pts_prev;
        cv::Mat desc_prev;
        float mMth; // good match ratio threshold
    };
}