#include"Generator.h"
#include<pthread.h>     // 使用pthread库
#include<cstdio>
#include<cstdlib>
#include<chrono>        // 用来计时，不用clock()是因为其计算的是处理器的时间，不是实际时间

vector<double> A, B, C;     // 矩阵A、B、C
int M, N, K, num, cols_per_thread;
// 矩阵的大小信息(M，N，K)、线程数量num、每个线程要计算的矩阵B的列数cols_per_thread

// 打印矩阵
void printMatrix(vector<double> mat, int rows, int cols){
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
           printf("%lf ", mat[i * cols + j]);
        }
        printf("\n");
    }
    return;
}


void* matrix_multiply(void* id){
    int thread_id = *((int*)id);    // 获取当前线程号
    int begin_Col = thread_id * cols_per_thread, end_Col = (thread_id + 1) * cols_per_thread;
    // 根据线程号获取当前线程矩阵B需要参与运算的起始列数和终止列数
    // 进行矩阵乘法
    for(int i = 0; i < M; ++i){
        for(int k = begin_Col;k < end_Col; ++k){
            double temp = 0;
            for(int j = 0; j < N; ++j){
                temp += A[i*N + j] * B[j*K + k];
            }
            C[i*K + k] = temp;
        }
    }
    pthread_exit(NULL);     // 终止线程

}
int main(int argc, char* argv[]){
 // 在运行时直接在最后输入参数：M，N，K，num(线程数量)
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    num = atoi(argv[4]);

    const int THREAD_NUM = num;
    cols_per_thread = K / THREAD_NUM;   // 计算每个线程中矩阵B有多少列要参与运算
    // 初始化矩阵A、B、C并将A、B打印出来
    A.resize(M * N);
    B.resize(N * K);
    C.resize(M * K);
    
    A = random_matrix_generator(M, N);
    B = random_matrix_generator(N, K);
    printf("Matrix A: \n");
    printMatrix(A, M, N);
    printf("Matrix B: \n");
    printMatrix(B, N, K);

    pthread_t handles[THREAD_NUM];  // 为线程分配空间
    int thread_ids[THREAD_NUM];     // 线程号

   auto start_time = chrono::high_resolution_clock::now();     // 记录开始时间
    for(int i = 0; i < THREAD_NUM; ++i){
        thread_ids[i] = i;
        pthread_create(&handles[i], NULL, matrix_multiply, (void*)&thread_ids[i]);   // 创建并启动线程
    }

    for(int i = 0; i < THREAD_NUM; ++i){
        pthread_join(handles[i], NULL);    // 等待线程结束并回收线程资源
    }

    if(K % THREAD_NUM != 0){
        // 可能M未必能被THREAD_NUM整除，因此要计算剩余的(M % THREAD_NUM)行
        for(int i = 0; i < M; ++i){
            for(int k = THREAD_NUM * cols_per_thread; k < K; ++k){
                double temp = 0;
                for(int j = 0; j < N; ++j){
                    temp += A[i*N +j] * B[j*K + k];
                }
                C[i*K + k] = temp;
            }
        }
    }
   auto end_time = chrono::high_resolution_clock::now();   // 记录结束时间
   auto using_time = (double)(end_time - start_time).count() / 1e9;  
    /*
        使用chrono::high_resolution_clock::now()算出来的时间差(chrono::high_resolution_clock::duration<double>)是纳
        秒级别的，所以要除以1e9也即10的9次方才能转换成秒
    */
   
   // 打印矩阵C
    printf("Matrix C: \n");
    printMatrix(C, M, K);   
    
    // 打印使用pthread多线程实现并行矩阵乘法所用时间
    printf("在%d个线程并行的情况下计算大小为%d*%d的矩阵A和大小为%d*%d的矩阵B所用时间为: %lf s\n", THREAD_NUM,M,N,N,K,using_time);
    return 0;
}
