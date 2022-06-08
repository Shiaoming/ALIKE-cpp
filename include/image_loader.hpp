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

#include <glob.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include <utils.h>

namespace alike
{
    class ImageLoader
    {
    public:
        ImageLoader(std::string filepath)
        {
            if (startsWith(filepath, "camera"))
            {
                auto camera = atoi(filepath.substr(6, filepath.size()).c_str());
                mCap = cv::VideoCapture(camera);
                if (!mCap.isOpened())
                {
                    std::cout << "can't open " << camera << std::endl;
                    exit(-1);
                }
                mMode = CAMERA;
            }
            else
            {
                if (isFileExist(filepath))
                {
                    if (isFile(filepath))
                    {
                        mCap = cv::VideoCapture(filepath);
                        if (!mCap.isOpened())
                        {
                            std::cout << "can't open " << filepath << std::endl;
                            exit(-1);
                        }
                        auto fps = mCap.get(cv::CAP_PROP_FPS);
                        auto len = mCap.get(cv::CAP_PROP_FRAME_COUNT) - 1;
                        auto duration = len / fps;
                        std::cout << "Opened video " << filepath << std::endl;
                        std::cout << "Frames: " << len << ", FPS: " << fps << ", Duration: " << duration << "s" << std::endl;

                        mMode = VIDEO;
                    }
                    else if (isDir(filepath))
                    {
                        std::vector<std::string> filenames;
                        getFileNames(filepath, filenames);

                        for (auto &filename : filenames)
                        {
                            if (endsWith(filename, ".png") || endsWith(filename, ".jpg") || endsWith(filename, ".ppm"))
                            {
                                mImageList.push_back(filename);
                            }
                        }
                        std::sort(mImageList.begin(), mImageList.end());
                        std::cout << "Loading " << mImageList.size() << " images from " << filepath << std::endl;
                        mMode = IMAGES;
                    }
                    else
                    {
                        std::cout << "Incorrect input: " << filepath << std::endl;
                        std::cout << ">> input should be: camerax/path of images/path of videos" << std::endl;
                        exit(-1);
                    }
                }
                else
                {
                    std::cout << filepath << "dosen't exists!" << std::endl;
                    exit(-1);
                }
            }
        }

        bool read_next(cv::Mat &frame, int max_size = -1)
        {
            bool ret = true;
            switch (mMode)
            {
            case CAMERA:
            case VIDEO:
                ret = mCap.read(frame);
                if (!ret)
                    std::cout << "Can't receive frame (stream end?). Exiting ..." << std::endl;
                break;
            case IMAGES:
                if (mIdx >= mImageList.size())
                {
                    std::cout << "End of images." << std::endl;
                    ret = false;
                }
                else
                {
                    auto image_name = mImageList[mIdx++].c_str();
                    frame = cv::imread(image_name);
                    if (frame.data == 0)
                    {
                        std::cout << "Can't read " << image_name << " Exiting ..." << std::endl;
                        ret = false;
                    }
                }
                break;
            default:
                ret = false;
                break;
            }

            if (ret && max_size > 0)
            {
                auto H0 = frame.rows;
                auto W0 = frame.cols;
                if (W0 > H0)
                    cv::resize(frame, frame, cv::Size(max_size, int(H0 * max_size / W0)));
                else
                    cv::resize(frame, frame, cv::Size(int(W0 * max_size / H0), max_size));
            }

            return ret;
        }

    private:
        cv::VideoCapture mCap;
        enum MODE
        {
            CAMERA,
            VIDEO,
            IMAGES
        } mMode;
        std::vector<std::string> mImageList;
        int mIdx = 0;
    };
}