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
#include <chrono>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <args.hxx>
#include <image_loader.hpp>
#include <alike.hpp>
#include <simple_tracker.hpp>
#include <utils.h>
#include <iostream>
#include <sstream>

using std::stringstream;

using namespace alike;

int main(const int argc, char *argv[])
{
    // ===============> args
    args::ArgumentParser parser("ALIKE-demo");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::Positional<std::string> file_path_parser(parser,
                                                   "input",
                                                   "Image directory or movie file or 'camera0' (for webcam0).",
                                                   args::Options::Required);
    args::Positional<std::string> model_path_parser(parser,
                                                    "model",
                                                    "Path of alike torchscript model.",
                                                    args::Options::Required);

    args::Flag use_cuda_parser(parser, "cuda", "Use cuda or not.", {"cuda"});
    args::ValueFlag<int> top_k_parser(parser,
                                      "top_k",
                                      "Detect top K keypoints. <=0 for threshold based mode, >0 for top K mode. [default: -1]",
                                      {"top_k"},
                                      -1);
    args::ValueFlag<float> scores_th_parser(parser,
                                            "scores_th",
                                            "Detector score threshold. [default: 0.2]",
                                            {"scores_th"},
                                            0.2);
    args::ValueFlag<int> n_limit_parser(parser,
                                        "n_limit",
                                        "Maximum number of keypoints to be detected. [default: 5000]",
                                        {"n_limit"},
                                        5000);
    args::ValueFlag<float> ratio_parser(parser,
                                        "ratio",
                                        "Ratio in FLANN matching process. [default: 0.7]",
                                        {"ratio"},
                                        0.7);
    args::ValueFlag<int> max_size_parser(parser,
                                         "max_size",
                                         "Maximum image size. (<=0 original; >0 for maximum image size). [default: -1]",
                                         {"max_size"},
                                         -1);
    args::Flag no_display_parser(parser,
                                 "no_display",
                                 "Do not display images to screen. Useful if running remotely.",
                                 {"no_display"});
    args::Flag no_subpixel_parser(parser,
                                  "no_subpixel",
                                  "Do not detect sub-pixel keypoints.",
                                  {"no_subpixel"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    std::string file_path = args::get(file_path_parser);
    std::string model_path = args::get(model_path_parser);
    bool use_cuda = args::get(use_cuda_parser);
    int top_k = args::get(top_k_parser);
    float scores_th = args::get(scores_th_parser);
    int n_limit = args::get(n_limit_parser);
    int max_size = args::get(max_size_parser);
    float ratio = args::get(ratio_parser);
    bool no_display = args::get(no_display_parser);
    bool no_subpixel = args::get(no_subpixel_parser);

    std::cout << "=======================" << std::endl;
    std::cout << "Running with " << ((use_cuda) ? "CUDA" : "CPU") << "!" << std::endl;
    std::cout << "=======================" << std::endl;

    // ===============> create model
    auto loader = ImageLoader(file_path);
    auto alike = ALIKE(model_path, use_cuda, 2, top_k, scores_th, n_limit, !no_subpixel);
    auto tracker = SimpleTracker();

    if (!no_display)
    {
        std::cout << "Press 'q' to stop!" << std::endl;
        cv::namedWindow("win");
    }

    // ===============> main loop
    cv::Mat image;
    std::vector<int> runtimes;
    auto device = (use_cuda) ? torch::kCUDA : torch::kCPU;
    while (loader.read_next(image, max_size))
    {
        torch::Tensor score_map, descriptor_map;
        torch::Tensor keypoints_t, dispersitys_t, kptscores_t, descriptors_t;
        std::vector<cv::KeyPoint>
            keypoints;
        cv::Mat descriptors;
        cv::Mat img_rgb;
        cv::cvtColor(image, img_rgb, cv::COLOR_BGR2RGB);
        auto img_tensor = mat2Tensor(image).permute({2, 0, 1}).unsqueeze(0).to(device).to(torch::kFloat) / 255;

        // core
        using namespace std::chrono;
        if (use_cuda)
            torch::cuda::synchronize();
        auto start = system_clock::now();
        // ====== core
        alike.extract(img_tensor, score_map, descriptor_map);
        alike.detectAndCompute(score_map, descriptor_map, keypoints_t, dispersitys_t, kptscores_t, descriptors_t);
        // ====== core
        if (use_cuda)
            torch::cuda::synchronize();
        auto end = system_clock::now();
        milliseconds mill = duration_cast<milliseconds>(end - start);
        runtimes.push_back(mill.count());

        // Note: for a keypoint of cv::KeyPoint
        // keypoint.size=dispersity is the dispersity
        // keypoint.response=score is the score
        alike.toOpenCVFormat(keypoints_t, dispersitys_t, kptscores_t, descriptors_t, keypoints, descriptors);
        cv::Mat track_img = image.clone();
        auto N_matches = tracker.update(track_img, keypoints, descriptors);

        // get fps
        float ave_fps = 0;
        for (auto i = 0; i < runtimes.size(); i++)
            ave_fps += 1000 / runtimes[i];
        ave_fps = ave_fps / runtimes.size();
        stringstream fmt;
        fmt << "FPS: " << ave_fps << ", Keypoints/Matches: " << keypoints.size() << "/" << N_matches;
        std::string status = fmt.str();
        std::cout << status << std::endl;

        // visualization
        if (!no_display)
        {
            auto score_map_mat = tensor2Mat(score_map);
            auto scoremap_jet = applyJet(score_map_mat);

            cv::setWindowTitle("win", status);
            cv::imshow("win", track_img);
            cv::imshow("scoremap", scoremap_jet);
            auto c = cv::waitKey(1);
            if (c == 'q')
                break;
        }
    }

    std::cout << "Finished!" << std::endl;
    if (!no_display)
    {
        std::cout << "Press any key to exit!" << std::endl;
        cv::waitKey();
    }
    return 0;
}