#include"cuda_runtime.h"
#include<cstdio>
#include<fstream>
#include<random>
using namespace std;

__host__ void init(float* mat, int row, int col) {
    // 初始化随机数生成器
    random_device rd;
    default_random_engine eng(rd());
    uniform_int_distribution<int> distr(0, 255);

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            mat[i * col + j] = distr(eng);
        }
    }
}

// 保存卷积后的矩阵
__host__ void write_result(float* r, float* g, float *b, int row, int col, string filename){
    ofstream f(filename);
    if(f.is_open()){
        f << "red:\n";
        for(int i = 0; i < row; ++i){
            for(int j = 0; j < col; ++j){
                if(r[i * col + j] > 255){
                    r[i * col + j] = 255;
                }
                if(r[i * col + j] < 0){
                    r[i * col + j] = 0;
                }
                f << r[i * col + j]<<" ";
            }
            f << "\n";
        }
        f << "\n" << "green:\n";
        for(int i = 0; i < row; ++i){
            for(int j = 0; j < col; ++j){
                if(g[i * col + j] > 255){
                    g[i * col + j] = 255;
                }
                if(g[i * col + j] < 0){
                    g[i * col + j] = 0;
                }                
                f << g[i * col + j]<<" ";
            }
            f << "\n";
        }
        f << "\n" << "blue:\n";
        for(int i = 0; i < row; ++i){
            for(int j = 0; j < col; ++j){
                if(b[i * col + j] > 255){
                    b[i * col + j] = 255;
                }
                if(b[i * col + j] < 0){
                    b[i * col + j] = 0;
                }                
                f << b[i * col + j]<<" ";
            }
            f << "\n";
        }
    }
}

// image to column
__global__ void im2col(float* input, float* output, int f_w, int f_h, int w_in, int h_in, int w_out, int h_out, int s){
    int y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= 0 && x < w_out && y >= 0 && y < h_out){
        int col_index = y * w_out + x;
        for(int k = 0; k < f_h; ++k){
            for(int p = 0; p < f_w; ++p){
                int row_index = k * f_w + p;
                int input_row = y * s + k, input_col = x * s + p;
                output[row_index * (w_out * h_out) + col_index] = input[input_row * w_in + input_col];
            }
         }
    }
}

// 卷积
__global__ void convolution_shared(float* input, float* output, float* filter, int w_out, int h_out, int f_w, int f_h){
    extern __shared__ float smem[];
    int y = blockIdx.y * blockDim.y + threadIdx.y, x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= 0 && x < w_out && y >= 0 && y < h_out){
        int index_1 = y * w_out + x;
        for(int i = 0; i <f_h; ++i){
            for(int j = 0; j < f_w; ++j){
                smem[i * f_w + j] = input[(i * f_w + j) * (w_out * h_out) + index_1];
            }
        }
        __syncthreads();
        float sum = 0;
        for(int i = 0; i < f_h; ++i){
            for(int j = 0; j < f_w; ++j){
                int index_2 = i * f_w + j;
                sum += smem[index_2] * filter[index_2];
            }
        }
        atomicAdd(&output[index_1], sum);
    }
}

int main(){
    int w, h, w_pad, h_pad, w_out, h_out, f_w = 3, f_h = 3, s, threads, blockDim_x = 1, blockDim_y = 1;
    printf("请输入图像大小：\n");
    scanf("%d %d", &h, &w);
    
    printf("请输入步长：\n");
    scanf("%d", &s);
    
    printf("请输入输出图像大小：\n");
    scanf("%d %d", &h_out, &w_out);
    
    printf("请输入每个线程块内的线程数：\n");
    scanf("%d", &threads);
    
    printf("请输入线程块的维度：\n");
    while (scanf("%d %d", &blockDim_y, &blockDim_x) == 2 && blockDim_y * blockDim_x != threads) {
        printf("输入的维度不符合要求，两个维度的乘积要等于每个块内的线程数，请重新输入：\n");
    }
    float *r, *g, *b, *r_pad, *g_pad, *b_pad, *r_res, *g_res, *b_res, *f_1, *f_2, *f_3, *r_col, *g_col, *b_col;
    cudaMallocHost(&r, sizeof(float) * w * h);
    cudaMallocHost(&g, sizeof(float) * w * h);
    cudaMallocHost(&b, sizeof(float) * w * h);
    
    // 获取原始的图像三通道的像素矩阵
    init(r, h, w);
    init(g, h, w);
    init(b, h, w);
    
    int delta_w = (w_out - 1) * s - w + f_w, delta_h = (h_out - 1) * s - h + f_h;//根据输出维度、步长和卷积核大小，计算出需要额外填充的列数和行数
    bool flag = (delta_w != 0) && (delta_h != 0); //行和列是否都需要填充(由于宽高相等，因此行和列要么都要进行填充，要么都不需要进行填充)
    if(flag){
        // 计算上下左右外围需要填充的行数和列数，如果能被2整除，上下/左右各填充一半，如果不行则下方/右侧多填充一行/列
        int pad_half_w_1 = delta_w / 2, pad_half_w_2 = (delta_w % 2 == 0) ? delta_w / 2 : delta_w / 2 + 1;
        int pad_half_h_1 = delta_h / 2, pad_half_h_2 = (delta_h % 2 == 0) ? delta_h / 2 : delta_h / 2 + 1;
        w_pad = w + delta_w;
        h_pad = h + delta_h;
        cudaMallocHost(&r_pad, sizeof(float) * w_pad * h_pad);
        cudaMallocHost(&g_pad, sizeof(float) * w_pad * h_pad);
        cudaMallocHost(&b_pad, sizeof(float) * w_pad * h_pad);
        for(int i = 0; i < h_pad; ++i){
            for(int j = 0; j < w_pad; ++j){
                int index = i * w_pad + j;
                if(i >= pad_half_h_1 && i < h_pad - pad_half_h_2 &&
                   j >= pad_half_w_1 && j < w_pad - pad_half_w_2){
                    int original_index = (i - pad_half_h_1) * w + (j - pad_half_w_1);
                    r_pad[index] = r[original_index];
                    g_pad[index] = g[original_index];
                    b_pad[index] = b[original_index];
                }
                else{
                    r_pad[index] = 0;
                    g_pad[index] = 0;
                    b_pad[index] = 0;
                }
            }
        }
        
    }
    else{//不用进行零填充,直接将前面的r、g、b复制给r_pad、g_pad、b_pad即可
        w_pad = w;
        h_pad = h;
        cudaMallocHost(&r_pad, sizeof(float) * w_pad * h_pad);
        cudaMallocHost(&g_pad, sizeof(float) * w_pad * h_pad);
        cudaMallocHost(&b_pad, sizeof(float) * w_pad * h_pad);
        cudaMemcpy(r_pad, r, w_pad * h_pad * sizeof(float), cudaMemcpyHostToHost);
        cudaMemcpy(g_pad, g, w_pad * h_pad * sizeof(float), cudaMemcpyHostToHost);
        cudaMemcpy(b_pad, b, w_pad * h_pad * sizeof(float), cudaMemcpyHostToHost);
    }
    
    // 初始化3个三通道卷积核
    cudaMallocHost(&f_1, sizeof(float) * f_h * f_w * 3);
    cudaMallocHost(&f_2, sizeof(float) * f_h * f_w * 3);
    cudaMallocHost(&f_3, sizeof(float) * f_h * f_w * 3);
    for(int i = 0; i < f_h * f_w * 3; ++i){
        f_1[i] = 0.01;
        f_2[i] = 0.05;
        f_3[i] = 0.09;
    }
    
    dim3 blockDim(blockDim_x, blockDim_y, 1);
    dim3 gridDim((int)((w_out + blockDim.x - 1)/blockDim.x), (int)((h_out + blockDim.y - 1)/blockDim.y));
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaMallocHost(&r_res, sizeof(float) * h_out * w_out);
    cudaMallocHost(&g_res, sizeof(float) * h_out * w_out);
    cudaMallocHost(&b_res, sizeof(float) * h_out * w_out);
    cudaMallocHost(&r_col, sizeof(float) * h_out * w_out * f_w * f_h);
    cudaMallocHost(&g_col, sizeof(float) * h_out * w_out * f_w * f_h);
    cudaMallocHost(&b_col, sizeof(float) * h_out * w_out * f_w * f_h);
    cudaEventRecord(start, 0);
    im2col<<<gridDim, blockDim>>>(r_pad, r_col, f_w, f_h, w_pad, h_pad, w_out, h_out, s);
    im2col<<<gridDim, blockDim>>>(g_pad, g_col, f_w, f_h, w_pad, h_pad, w_out, h_out, s);
    im2col<<<gridDim, blockDim>>>(b_pad, b_col, f_w, f_h, w_pad, h_pad, w_out, h_out, s);
    convolution_shared<<<gridDim, blockDim, sizeof(float) * f_w * f_h>>>(r_col, r_res, f_1, w_out, h_out, f_w, f_h);
    convolution_shared<<<gridDim, blockDim, sizeof(float) * f_w * f_h>>>(g_col, g_res, f_1+9, w_out, h_out, f_w, f_h);
    convolution_shared<<<gridDim, blockDim, sizeof(float) * f_w * f_h>>>(b_col, b_res, f_1+18, w_out, h_out, f_w, f_h);
    convolution_shared<<<gridDim, blockDim, sizeof(float) * f_w * f_h>>>(r_col, r_res, f_2, w_out, h_out, f_w, f_h);
    convolution_shared<<<gridDim, blockDim, sizeof(float) * f_w * f_h>>>(g_col, g_res, f_2+9, w_out, h_out, f_w, f_h);
    convolution_shared<<<gridDim, blockDim, sizeof(float) * f_w * f_h>>>(b_col, b_res, f_2+18, w_out, h_out, f_w, f_h);
    convolution_shared<<<gridDim, blockDim, sizeof(float) * f_w * f_h>>>(r_col, r_res, f_3, w_out, h_out, f_w, f_h);
    convolution_shared<<<gridDim, blockDim, sizeof(float) * f_w * f_h>>>(g_col, g_res, f_3+9, w_out, h_out, f_w, f_h);
    convolution_shared<<<gridDim, blockDim, sizeof(float) * f_w * f_h>>>(b_col, b_res, f_3+18, w_out, h_out, f_w, f_h);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);
    printf("图像大小:%d*%d;线程块维度:(%d,%d);访存方式:共享内存;所用时间:%f ms\n",h,w,blockDim_y,blockDim_x,ms);
    
    string output_filename = to_string(h) + "*" + to_string(w) + " " + "(" + to_string(blockDim_y) + "," +  to_string(blockDim_x) + ") im2col shared.txt";
    write_result(r_res, g_res, b_res, h_out, w_out, output_filename);
    cudaFreeHost(r);
    cudaFreeHost(g);
    cudaFreeHost(b);
    cudaFreeHost(r_res);
    cudaFreeHost(g_res);
    cudaFreeHost(b_res);
    cudaFreeHost(r_pad);
    cudaFreeHost(g_pad);
    cudaFreeHost(b_pad);
    cudaFreeHost(r_col);
    cudaFreeHost(g_col);
    cudaFreeHost(b_col);
    cudaFreeHost(f_1);
    cudaFreeHost(f_2);
    cudaFreeHost(f_3);
    return 0;
}