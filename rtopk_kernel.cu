#include "rtopk_kernel.cuh"

template <int WARPS_PER_BLOCK>
__global__ void rtopk_kernel(float *data, float *value, int *index, int N, int dim_origin, int k, int max_iter, float precision)
{
    extern __shared__ float cache[]; 
    const int wid = threadIdx.x / 32;
    const int laneid = threadIdx.x % 32;

    if (blockIdx.x * WARPS_PER_BLOCK + wid >= N){
        return;
    }

    const int dim_len = (dim_origin + 31) / 32;

    const int first_idx = blockIdx.x * WARPS_PER_BLOCK * dim_origin;
    #pragma unroll
    for(int ext = 0; ext < dim_len; ext++){
        int data_idx = wid * dim_origin + laneid + ext * 32;
        if (data_idx < dim_origin){
            cache[data_idx] = data[first_idx + data_idx];
        }
    }

    __syncwarp();

    float max_data = -99999, min_data = 99999;

    #pragma unroll
    for(int j = 0; j < dim_len; j++){
        int data_idx = wid * dim_origin + laneid  + j * 32;
        if (data_idx >= dim_origin) {
            break;
        }
        if(cache[data_idx] > max_data){
            max_data = cache[data_idx];
        }
        if(cache[data_idx] < min_data){
            min_data = cache[data_idx];
        }
    }

    #pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        max_data = max(max_data, __shfl_down_sync(0xFFFFFFFF, max_data, offset));
        min_data = min(min_data, __shfl_down_sync(0xFFFFFFFF, min_data, offset));
    }

    max_data = __shfl_sync(0xFFFFFFFF, max_data, 0);
    min_data = __shfl_sync(0xFFFFFFFF, min_data, 0);

    float mid_data = max_data;

    int count;

    for(int i = 0; ; i++){
        count = 0;
        #pragma unroll
        for(int j = 0; j < dim_len; j++){
            int data_idx = wid * dim_origin + laneid  + j * 32;
            if (data_idx < dim_origin) {
                count += cache[data_idx] >= mid_data;
            }
        }
        count += __shfl_down_sync(0xffffffff, count, 16);
        count += __shfl_down_sync(0xffffffff, count, 8);
        count += __shfl_down_sync(0xffffffff, count, 4);
        count += __shfl_down_sync(0xffffffff, count, 2);
        count += __shfl_down_sync(0xffffffff, count, 1);
        count = __shfl_sync(0xffffffff, count, 0);

        if(i >= max_iter || mid_data <= min_data + precision){
            break;
        }

        if(count < k){
            max_data = mid_data;
        }
        else if(count > k){
            min_data = mid_data;
        }
        else{
            break;
        }        
        mid_data = (min_data + max_data) / 2;        
    }

    int eq_n = k - count; 
    int total_cnt = 0, total_cnt_eq = 0, total_cnt_whole = 0;

    #pragma unroll
    for(int ext = 0; ext < dim_len; ext++){
        if(total_cnt_whole >= k){
            break;
        }
        int data_idx = wid * dim_origin + laneid + ext * 32;
        if (data_idx >= dim_origin) {
            break;
        }
        float val = cache[data_idx];
        bool choose = val >= mid_data;

        bool choose_eq = val >= min_data && val < mid_data;

        unsigned mask = __ballot_sync(0xffffffff, choose); 
        unsigned mask_eq = __ballot_sync(0xffffffff, choose_eq);

        int lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));
        int lane_cnt_eq = __popc(mask_eq & ((1 << (laneid + 1)) - 1));

        if (total_cnt + lane_cnt > k) {
            choose = 0;
        }       
        if (total_cnt_eq + lane_cnt_eq > eq_n ){
            choose_eq = 0;
        }

        mask = __ballot_sync(0xffffffff, choose);
        mask_eq = __ballot_sync(0xffffffff, choose_eq);

        unsigned mask_whole = mask | mask_eq;

        lane_cnt = __popc(mask & ((1 << (laneid + 1)) - 1));
        lane_cnt_eq = __popc(mask_eq & ((1 << (laneid + 1)) - 1));
        int lane_cnt_whole = __popc(mask_whole & ((1 << (laneid + 1)) - 1));

        if(choose || choose_eq){
            value[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt_whole + lane_cnt_whole - 1] = val;
            index[blockIdx.x * WARPS_PER_BLOCK * k + wid * k + total_cnt_whole + lane_cnt_whole - 1] = laneid + ext * 32;
        }

        total_cnt += lane_cnt;
        total_cnt = __shfl_sync(0xffffffff, total_cnt, 31);

        total_cnt_eq += lane_cnt_eq;
        total_cnt_eq = __shfl_sync(0xffffffff, total_cnt_eq, 31);

        total_cnt_whole += lane_cnt_whole;
        total_cnt_whole = __shfl_sync(0xffffffff, total_cnt_whole, 31);

    }
}


template __global__ void rtopk_kernel<8>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_kernel<4>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_kernel<2>(float*, float*, int*, int, int, int, int, float);
template __global__ void rtopk_kernel<1>(float*, float*, int*, int, int, int, int, float);