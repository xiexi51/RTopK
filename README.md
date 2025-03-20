# RTopK
Official Implementation of "RTop-K: Ultra-Fast Row-Wise Top-K Selection for Neural Network Acceleration on GPUs" 

Please cite our paper if you use the code ✔
```
@inproceedings{
  xie2025rtopk,
  title={{RT}op-K: Ultra-Fast Row-Wise Top-K Selection for Neural Network Acceleration on {GPU}s},
  author={Xi Xie and Yuebo Luo and Hongwu Peng and Caiwen Ding},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=PHg4rAXFVH}
}
```


## Clone and build the project
```sh
git clone https://github.com/xiexi51/RTopK.git
cd RTopK
mkdir build
cd build
cmake ..
make
```

## Run the test
```sh
./rtopk
```

## Requirements
- CUDA Toolkit (>= 12.0)
- CMake (>= 3.5)
- C++ Standard (>= 17)

## License
This project is licensed under the MIT License.
