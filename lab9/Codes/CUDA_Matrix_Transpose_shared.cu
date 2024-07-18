#include<cstdio>
#include<cuda_runtime.h>
#include<cstring>
#include<cmath>

#define N 2048
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
#define TILE_SIZE 32
// 随机初始化size*size的矩阵
__host__ void init(int* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = rand() % 10;
        }
    }
}

// 打印size行size列的矩阵
__host__ void printMatrix(int* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%d ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

// 使用共享内存进行矩阵转置
__global__ void transpose_shared(int* original, int* transposed, int size) {
    __shared__ int smem[TILE_SIZE][TILE_SIZE + 1];//避免bank冲突，设置TILE_SIZE+1列
    int bx = blockIdx.x * TILE_SIZE, by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = bx + tx, y = by + ty;

    if (x < size && y < size) {
        smem[ty][tx] = original[y * size + x];
    }
    __syncthreads();
    if (bx + ty < size && by + tx < size) {
        transposed[(bx + ty) * size + (by + tx)] = smem[tx][ty];
    }
}

int main(){
    int *original_matrix, *transposed_matrix;
    cudaMallocHost((void**)&original_matrix, sizeof(int) * N * N); // 使用 cudaMallocHost 分配原矩阵内存
    cudaMallocHost((void**)&transposed_matrix, sizeof(int) * N * N); // 使用 cudaMallocHost 分配转置后的矩阵内存
    init(original_matrix, N);   // 初始化原矩阵
    
    printf("原矩阵:\n");
    printMatrix(original_matrix,N);
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 dimGrid((N+BLOCK_DIM_X-1)/BLOCK_DIM_X, (N+BLOCK_DIM_Y-1)/BLOCK_DIM_Y, 1);
    
    //创建事件
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaEventRecord(start, 0); // 记录事件
    transpose_shared<<<dimGrid, dimBlock>>>(original_matrix, transposed_matrix, N);
    cudaEventRecord(end, 0); // 记录事件
    cudaEventSynchronize(end); // 同步
    float shared_time = 0;
    cudaEventElapsedTime(&shared_time, start, end); // 计时
    
    
    printf("使用共享内存进行转置后的矩阵:\n");
    printMatrix(transposed_matrix,N);
    printf("线程块大小:(%d,%d) 矩阵规模:%d*%d 访存方式:共享内存 using time:%f ms\n", BLOCK_DIM_X,BLOCK_DIM_Y,N,N,shared_time);
    
    // 释放内存
    cudaFreeHost(original_matrix);
    cudaFreeHost(transposed_matrix);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    
    return 0;
    
    
}