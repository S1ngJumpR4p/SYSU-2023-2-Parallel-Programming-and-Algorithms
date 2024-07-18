#include "Parallel_for.h"
#include <pthread.h>


void printMatrix(vector<double>& mat, int row, int col){
    for(int i = 0; i < row; ++i){
        for(int j = 0; j < col; ++j){
            printf("%lf ",mat[i*col+j]);
        }
        printf("\n");
    }
}

void* functor( void* args){
    FUNCTOR_ARGS* Args = (FUNCTOR_ARGS*)args;
    int N = Args->N, K = Args->K;

    // 通用矩阵乘法
    for(int i = 0; i < Args->Indices.size(); ++i){
        int m = Args->Indices[i];   // 获取A要参与计算的行号
        for(int k = 0; k < K; ++k){
            for(int n = 0; n < N; ++n){
                C[m*K + k] += A[m*N + n] * B[n*K + k];
            }
        }
    }
    pthread_exit(NULL);
}

double parallel_for(int start, int end, int inc, void* (*functor_ptr)(void*), void* args, int thread_num){
    pthread_t* handles = new pthread_t[thread_num];
    FUNCTOR_ARGS** missions = (FUNCTOR_ARGS**)args;
    auto start_time = chrono::high_resolution_clock::now();     // 开始计时
    for(int i = 0; i <thread_num; ++i){
        pthread_create(&handles[i],NULL, functor_ptr, (void*)missions[i]);  // 创建并启动线程
    }
    for(int i = 0; i < thread_num; ++i){
        pthread_join(handles[i],NULL);      // 等待线程结束并释放资源
    }
    auto end_time = chrono::high_resolution_clock::now();   // 结束计时
    auto using_time = (end_time-start_time).count()/1e9;
    return using_time;
}
