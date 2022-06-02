# CPP implementation of [ALIKE](https://github.com/Shiaoming/ALIKE)

ALIKE applies a differentiable keypoint detection module to detect accurate sub-pixel keypoints. The network can run at 95 frames per second for 640 x 480 images on NVIDIA Titan X (Pascal) GPU and achieve equivalent performance with the state-of-the-arts. ALIKE benefits real-time applications in resource-limited platforms/devices. Technical details are described in [this paper](https://arxiv.org/pdf/2112.02906.pdf).

> ```
> Xiaoming Zhao, Xingming Wu, Jinyu Miao, Weihai Chen, Peter C. Y. Chen, Zhengguo Li, "ALIKE: Accurate and Lightweight Keypoint
> Detection and Descriptor Extraction," IEEE Transactions on Multimedia, 2022.
> ```

![](https://raw.githubusercontent.com/Shiaoming/ALIKE/c773cec0c8367a22351e45aef9a62dec78317936/assets/alike.png)


If you use ALIKE in an academic work, please cite:

```
@article{Zhao2022ALIKE,
    title = {ALIKE: Accurate and Lightweight Keypoint Detection and Descriptor Extraction},
    url = {http://arxiv.org/abs/2112.02906},
    doi = {10.1109/TMM.2022.3155927},
    journal = {IEEE Transactions on Multimedia},
    author = {Zhao, Xiaoming and Wu, Xingming and Miao, Jinyu and Chen, Weihai and Chen, Peter C. Y. and Li, Zhengguo},
    month = march,
    year = {2022},
}
```



## 1. Prerequisites

The implementation mainly based on OpenCV and libtorch.

- [CUDA](https://developer.nvidia.com/cuda-toolkit)

- OpenCV
```shell
sudo apt install libopencv-dev
```

- libtorch

Download [libtorch](https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip) and unzip it to the repository root.

- OpenMP (Optional)


Tested environment:
```
GNU=9.3.0
NVCC=11.3.109
OpenMP=4.5
cuDNN=8.2.0
OpenCV=4.2.0
libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip
```


## 2. Build

Build the demo program.

```shell
mkdir build
cd build
cmake ..
make
```


## 3. Run demo

```shell
$ ./build/demo -h
  ./build/demo input model {OPTIONS}

    ALIKE-demo

  OPTIONS:

      -h, --help                        Display this help menu
      input                             Image directory or movie file or
                                        'camera0' (for webcam0).
      model                             Path of alike torchscript model.
      --cuda                            Use cuda or not.
      --top_k=[top_k]                   Detect top K keypoints. <=0 for
                                        threshold based mode, >0 for top K mode.
                                        [default: -1]
      --scores_th=[scores_th]           Detector score threshold. [default: 0.2]
      --n_limit=[n_limit]               Maximum number of keypoints to be
                                        detected. [default: 5000]
      --ratio=[ratio]                   Ratio in FLANN matching process.
                                        [default: 0.7]
      --no_display                      Do not display images to screen. Useful
                                        if running remotely.
      --no_subpixel                     Do not detect sub-pixel keypoints.
      "--" can be used to terminate flag options and force all following
      arguments to be treated as positional options
```



## 4. Examples

### KITTI example
```shell
./build/demo  assets/kitti models/alike-t.pt --cuda
```
![](https://raw.githubusercontent.com/Shiaoming/ALIKE/c773cec0c8367a22351e45aef9a62dec78317936/assets/kitti.gif)

### TUM example
```shell
./build/demo  assets/tum models/alike-t.pt --cuda
```
![](https://raw.githubusercontent.com/Shiaoming/ALIKE/c773cec0c8367a22351e45aef9a62dec78317936/assets/tum.gif)


For more details, please refer to the [paper](https://arxiv.org/abs/2112.02906).
