/*
使用MPI集合通信实现并行矩阵乘法
*/
#include <mpi.h>
#include "Generator.h"
#include <cstdio>
#include <cstdlib>

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

int main(int argc, char* argv[]){
    // 在运行时直接在最后输入参数：M，N，K，process_num(线程数量)
    int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]), process_num = atoi(argv[4]), myrank, rows_per_process;
    // myrank：进程号；rows_per_process：发送给除0号进程外的其他进程的矩阵A的行数

    vector<double> A(M * N), B(N * K), C(M * K,  0);    // 矩阵A、B、C，C=AB
    vector<double> temp_C(M * K);       // 用来存每个进程的计算结果
    double start_time, end_time;    // 记录并行矩阵乘法的开始时刻和结束时刻
    
    MPI_Init(&argc, &argv);     // MPI初始化
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);     // 获取通信子的进程数
    MPI_Comm_size(MPI_COMM_WORLD, &process_num);    // 获取进程在通信子中的编号
    rows_per_process = M / process_num;     // 每个进程计算的矩阵A分块的行数
    
    vector<double> temp_A(rows_per_process * N);     //用来存储矩阵A的分块

    if(myrank == 0){    // 0号进程

    // 调用随机矩阵生成函数random_matrix_generator，初始化矩阵A和B
        A = random_matrix_generator(M , N);    
        B = random_matrix_generator(N , K);

    // 输出矩阵A、B
        printf("Matrix A：\n");
        printMatrix(A, M, N);
        printf("\n");
        printf("Matrix B：\n");
        printMatrix(B, N, K);
        printf("\n");
        
        start_time = MPI_Wtime();   // 开始计时

        // 将矩阵B广播到其他进程
        MPI_Bcast(B.data(), N * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);   

        // 使用MPI_Scatter将矩阵A分块发送至其他进程
        MPI_Scatter(A.data(), rows_per_process * N, MPI_DOUBLE, temp_A.data(), rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // 由于MPI_Scatter也会发送矩阵A分块给0号进程，因此需要在0号进程也进行一次矩阵乘法
         for(int i = 0;i < rows_per_process; ++i){
            for(int k = 0; k < K; ++k){
                double tmp = 0;
                for(int j = 0; j < N; ++j){
                    tmp += temp_A[i * N + j] * B[j * K + k];
                }
                temp_C[i * K + k] = tmp;
            }
        }
        
        // 使用MPI_Gather汇总来自其他进程的运算结果
        MPI_Gather(temp_C.data(), rows_per_process * K, MPI_DOUBLE, C.data(), rows_per_process * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);   

        /* 把剩余部分算完，这是因为共有process_num个进程，使用MPI_Scatter将矩阵A的分块分给包括0号进程在内的所有进程后，考虑到M 不一定能被 process_num整除，
            因此需要计算余下的（ M % process_num ）行和矩阵B的乘积并存储于矩阵C的对应位置
        */
        for(int i = (process_num - 1) * rows_per_process; i < M; ++i){
            for(int k = 0; k < K; ++k){
                double tmp = 0;
                // 使用一个变量tmp是为了防止多线程环境下可能同时对 C[i * K + k]进行访问和修改，导致其值被覆盖，导致运算错误
                for(int j = 0; j < N; ++j){
                    tmp += A[i * N + j] * B[j * K + k];
                }
                C[i * K + k] = tmp;
            }
        }

        end_time = MPI_Wtime();     // 结束计时
        double delta_time = end_time - start_time;  // 求出矩阵乘法所用时间
        printf("Matrix C：\n");
        printMatrix(C, M, K);       // 输出矩阵C
        printf("\n");
        printf("\n在%d个进程并行的情况下计算大小为%d*%d的矩阵A与大小为%d*%d的矩阵B的乘积所用时间为：%lf s.\n", process_num, M, N, N, K, delta_time);
    }
    else{
         // 接收0号进程发过来的矩阵B
        MPI_Bcast(B.data(), N *K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // 接收0号进程使用MPI_Scatter散射发过来的矩阵A的分块
        MPI_Scatter(A.data(), rows_per_process * N, MPI_DOUBLE, temp_A.data(), rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);    

        // 用0号进程发送过来的矩阵A的分块和矩阵B进行通用矩阵乘法运算
        for(int i = 0; i < rows_per_process; ++i){
                for(int k = 0; k < K; ++k){
                    double tmp = 0;     
                    // 使用一个变量tmp是为了防止多线程环境下可能同时对 temp_C[i * K + k]进行访问和修改，导致其值被覆盖，导致运算错误               
                    for(int j = 0; j < N; ++j){
                        tmp += temp_A[i * N + j] * B[j * K + k];
                    }
                    temp_C[i * K + k] = tmp;
                }
        }

       MPI_Gather(temp_C.data(), rows_per_process * K, MPI_DOUBLE, C.data(), rows_per_process * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);   
        // 使用MPI_Gather将运算结果汇总到0号进程
    }
    MPI_Finalize();     // MPI结束
    return 0;
}