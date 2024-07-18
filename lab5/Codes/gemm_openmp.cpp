#include <omp.h>
#include "Generator.h"
#include <cstdio>

// 打印矩阵
void printMatrix(vector<double> mat, int row, int col){
    for(int i = 0; i < row; ++i){
        for(int j = 0; j < col; ++j){
            printf("%lf ",mat[i*col+j]);
        }
        printf("\n");
    }
}

// 使用静态调度方式进行并行矩阵乘法
void matrix_multiply_static(vector<double> A,vector<double> B,vector<double>& C, int thread_num, int M, int N, int K){
    C.resize(M*K);   // 为矩阵C分配空间
   #pragma omp parallel for num_threads(thread_num) schedule(static) shared(A,B,C)
        for(int i = 0; i < M; ++i){
            for(int k = 0; k < K; ++k){
                C[i*K+k] = 0;
                for(int j = 0; j < N; ++j){
                    C[i*K+k] += A[i*N+j] * B[j*K+k];
                }
            }
        }
}

// 使用动态调度方式进行并行矩阵乘法
void matrix_multiply_dynamic(vector<double> A,vector<double> B,vector<double>& C, int thread_num, int M, int N, int K){
    C.resize(M*K);  // 为矩阵C分配空间
   #pragma omp parallel for num_threads(thread_num) schedule(dynamic) shared(A,B,C)
        for(int i = 0; i < M; ++i){
            for(int k = 0; k < K; ++k){
                C[i*K+k] = 0;
                for(int j = 0; j < N; ++j){
                    C[i*K+k] += A[i*N+j] * B[j*K+k];
                }
            }
        }
}


int main(){
    int M, N, K, thread_num;    // 矩阵规模相关参数M、N、K(矩阵A大小：M*N；矩阵B大小：N*K)；线程数thread_num
    double start_time, end_time, using_time;    // 用于计时的参数
    printf("请输入矩阵规模参数M、N、K(矩阵A的大小为M*N，矩阵B的大小为N*K；在区间[128,2048]内的整数)：\n");
    scanf("%d %d %d",&M,&N,&K);
    printf("请输入线程数量(在区间[1,16]内的整数)：\n");
    scanf("%d",&thread_num);

    // 随机生成矩阵A和B
    vector<double> A = random_matrix_generator(M,N);
    vector<double> B = random_matrix_generator(N,K);

    // 打印矩阵A和B
    printf("Matrix A：\n");
    printMatrix(A,M,N);
    printf("Matrix B：\n");
    printMatrix(B,N,K);

    vector<double> C;

    /*接下来使用静态调度方式进行并行矩阵乘法*/
    start_time = omp_get_wtime();   // 开始计时
    matrix_multiply_static(A,B,C,thread_num,M,N,K);    
    end_time = omp_get_wtime();   // 结束计时
    using_time = end_time-start_time;      // 计算执行时间
    printf("Matrix C：\n"); 
    printMatrix(C,M,K);        // 打印矩阵C
    // 打印执行时间
    printf("在%d个线程并行的情况下，使用static调度方式计算大小为%d*%d的矩阵A乘%d*%d的矩阵B所用时间为：%lfs\n",thread_num,M,N,N,K,using_time);   

    /*接下来使用动态调度方式进行并行矩阵乘法*/
    start_time = omp_get_wtime();   // 开始计时
    matrix_multiply_dynamic(A,B,C,thread_num,M,N,K);    
    end_time = omp_get_wtime();   // 结束计时
    using_time = end_time-start_time;      // 计算执行时间
    // 打印执行时间
    printf("在%d个线程并行的情况下，使用dynamic调度方式计算大小为%d*%d的矩阵A乘%d*%d的矩阵B所用时间为：%lfs\n",thread_num,M,N,N,K,using_time); 

    return 0;
}