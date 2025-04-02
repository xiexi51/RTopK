#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <curand.h>
#include <algorithm>
#include <iomanip>
#include "rtopk_kernel.cuh"

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();
inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                          std::chrono::time_point<std::chrono::system_clock> b)
{
    return std::chrono::duration<double>(b - a).count();
}

using namespace std;

int main() {
    // int N_list[] = {16384, 65536, 262144, 1048576};
    int N_list[] = {65536};
    // int dim_origin_list[] = {128,256};
    int dim_origin_list[] = {8192};
    // int dim_k_list[] = {16, 32, 64, 96, 128};
    int dim_k_list[] = {64, 128, 256, 512};
    int max_iter_list[] = {2, 3, 4, 5, 6, 7, 8, 10000};
    // int max_iter_list[] = {10000};
    float precision_list[] = {0};

    
    int max_N = *std::max_element(std::begin(N_list), std::end(N_list));
    int max_dim_origin = *std::max_element(std::begin(dim_origin_list), std::end(dim_origin_list));
    int max_dim_k = *std::max_element(std::begin(dim_k_list), std::end(dim_k_list));

    cout << "max N = " << max_N << ", preparing data..." << endl;

    float *value;
    int *index;

    cudaMallocManaged(&value, max_N * max_dim_k * sizeof(float));
    cudaMallocManaged(&index, max_N * max_dim_k * sizeof(int));

    curandGenerator_t gen;
    float *devData;
    cudaMalloc((void **)&devData, max_N * max_dim_origin * sizeof(float));


    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, devData, max_N * max_dim_origin);
    
    cout << "data ready, testing..." << endl;

    ofstream fout("output3300.txt");

    for (int N : N_list){
        for (int dim_origin = 128; dim_origin <= 8192; dim_origin += 128){
        // for (int dim_origin : dim_origin_list){
            for (int dim_k : dim_k_list){
                if(dim_k >= dim_origin){
                    continue;
                }
                for (float precision : precision_list){
                    for (int max_iter : max_iter_list){
                        int w;
                        if (dim_origin <= 1024){
                            w = 8;
                        }
                        else if(dim_origin <= 2048){
                            w = 4;
                        }
                        else if(dim_origin <= 4096){
                            w = 2;
                        }
                        else{
                            w = 1;
                        }
                        int shared_size = w * dim_origin * sizeof(float);
                        int times = 4;
                        // warmup
                        for (int i = 0; i < times; i++) {
                            if(w == 8){
                                rtopk_kernel<8><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else if(w == 4){
                                rtopk_kernel<4><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else if(w == 2){
                                rtopk_kernel<2><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else{
                                rtopk_kernel<1><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                        }
                        cudaDeviceSynchronize();
                        double measured_time = 0;
                        for (int i = 0; i < times; i++) {
                            timestamp(t0);
                            if(w == 8){
                                rtopk_kernel<8><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else if(w == 4){
                                rtopk_kernel<4><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else if(w == 2){
                                rtopk_kernel<2><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            else{
                                rtopk_kernel<1><<<N / w, w * 32, shared_size>>>(devData, value, index, N, dim_origin, dim_k, max_iter, precision);
                            }
                            cudaDeviceSynchronize();
                            timestamp(t1);
                            measured_time += getDuration(t0, t1);
                        }

                        cout<< "N = " << N << ", dim_origin = " << dim_origin << ", dim_k = " << dim_k << ", max_iter = " << max_iter << ", topk time = " << measured_time / times * 1000 << endl;
                        fout<< "N = " << N << ", dim_origin = " << dim_origin << ", dim_k = " << dim_k << ", max_iter = " << max_iter << ", topk time = " << measured_time / times * 1000 << endl;
                        // fout.flush();
                    }
                }
            }
        }
    }

    fout.close();

    // Free unified memory
    cudaFree(value);
    cudaFree(index);
    curandDestroyGenerator(gen);
    cudaFree(devData);

    cout << "finish" << endl;

    return 0;
}