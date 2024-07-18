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

    vector<double> A(M * N), B(N * K), C(M * K);    // 矩阵A、B、C，C=AB
    vector<double> temp_C(M * K);       // 用来存每个非0号进程的计算结果
    double start_time, end_time;    // 记录并行矩阵乘法的开始时刻和结束时刻
    
    MPI_Init(&argc, &argv);     // MPI初始化
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);     // 获取通信子的进程数
    MPI_Comm_size(MPI_COMM_WORLD, &process_num);    // 获取进程在通信子中的编号
    rows_per_process = M / process_num;     // 每个进程计算的矩阵A分块的行数
    
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

        // 将矩阵B发送至其他进程
        for(int i = 1; i < process_num; ++i){
            MPI_Send(B.data(), N *K, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);     
        }

        // 将矩阵A分块发送至其他进程
        for(int i = 1; i < process_num; ++i){
            MPI_Send(A.data() + rows_per_process * (i-1) * N, rows_per_process * N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        }

        // 接收来自其他进程的运算结果
        for(int i = 1; i < process_num; ++i){
            MPI_Recv(temp_C.data(), rows_per_process * K, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j = 0; j < rows_per_process; ++j){
                for(int k = 0; k < K; ++k){
                    C[((i-1)*rows_per_process + j) * K + k] = temp_C[j * K + k];    // 将其他进程的运算结果(temp_C)存到矩阵C中
                }
            }
        }

        // 把剩余部分算完，这是因为共有process_num个进程，除了0号进程，剩余(process_num-1)个进程
        // 计算了(process_num-1)*(M/process_num)行，而A共有M行，因此还需要单独计算剩余的[M-(process_num-1)*(M/process_num)]行
        // 下为通用矩阵乘法
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
        MPI_Recv(B.data(), N * K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
        vector<double> temp_A(rows_per_process * N);     //用来存储矩阵A的分块

        // 接收0号进程发过来的矩阵A的分块
        MPI_Recv(temp_A.data(), rows_per_process * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    

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

        MPI_Send(temp_C.data(), rows_per_process * K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);    // 将运算结果发给0号进程
    }
    MPI_Finalize();     // MPI结束
    return 0;
}