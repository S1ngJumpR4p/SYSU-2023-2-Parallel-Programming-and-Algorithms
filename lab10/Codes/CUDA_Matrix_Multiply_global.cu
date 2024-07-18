#include<cstdio>
#include<cuda_runtime.h>
#include<random>

using namespace std;

// 随机初始化矩阵
__host__ void init(double* mat, int row, int col){
    // 初始化随机数生成器
    random_device rd;   
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(-5,5);
    
    for(int i = 0; i < row; ++i){
        for(int j = 0; j < col; ++j){
            mat[i*col+j] = distr(eng);
        }
    }
}

// 打印矩阵
__host__ void printMatrix(double* mat, int row, int col){
    for(int i = 0; i < row; ++i){
        for(int j = 0; j < col; ++j){
            printf("%lf ",mat[i*col+j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void matrix_multiply_global(double* A, double* B, double* C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;//计算行号
    int col = blockIdx.x * blockDim.x + threadIdx.x;//计算列号
    
    if(row < M && col < K){
        double value = 0;
        
        //计算A的第row行和B的第col列的乘积，将其存储在C的第row行第col列
        for(int i = 0; i < N; ++i){
            value += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = value;
    }
}

int main(){     
    int M,N,K,threads_per_block;
    printf("请输入M,N,K以及每个线程块的线程数:\n");
    scanf("%d %d %d %d",&M, &N, &K, &threads_per_block);
    double *A, *B, *C;
    
    // 为矩阵A,B,C分配空间
    cudaMallocHost((void**)&A, sizeof(double)*M*N);
    cudaMallocHost((void**)&B, sizeof(double)*N*K);
    cudaMallocHost((void**)&C, sizeof(double)*M*K);
    
    // 初始化矩阵A和B
    init(A,M,N);
    init(B,N,K);
    
    // 输出矩阵A和B
    printf("Matrix A:\n");
    printMatrix(A,M,N);
    printf("Matrix B:\n");
    printMatrix(B,N,K);
    
    cudaEvent_t start, end;//定义事件，用于计时
    
    //创建事件
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    int blockDim_x = 1, blockDim_y = 1;
    printf("请输入线程块的2个维度:\n");
    scanf("%d %d",&blockDim_x, &blockDim_y);
    while(blockDim_x * blockDim_y != threads_per_block){
        printf("输入的2个维度不符合要求(乘积要等于每个线程块的线程数),请重新输入:\n");
        scanf("%d %d",&blockDim_x, &blockDim_y);
    }
    
    dim3 dimBlock(blockDim_x, blockDim_y, 1);
    dim3 dimGrid((K + blockDim_x - 1) / blockDim_x, (M + blockDim_y - 1) / blockDim_y, 1);
    
    cudaEventRecord(start, 0);//开始计时
    matrix_multiply_global<<<dimGrid, dimBlock>>>(A, B, C, M, N, K);//调用核函数
    cudaEventRecord(end, 0);//结束计时
    cudaEventSynchronize(end);//同步
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);//计算耗时
    printf("Matrix C:\n");
    printMatrix(C, M, K);
    printf("线程块大小:(%d,%d) 矩阵A规模:%d*%d 矩阵B规模:%d*%d 访存方式:全局内存 所用时间:%f ms\n", blockDim_x, blockDim_y, M, N, N, K, ms);
    
    // 释放内存
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return 0;
}