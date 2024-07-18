#include "Generator.h"
#include "Parallel_for.h"
int M,N,K,thread_num,Inc, missions_per_thread;
set<int> unassigned;
vector<double> A, B, C;
int main(){
    printf("请输入矩阵规模参数M、N、K(矩阵A的大小为M*N，矩阵B的大小为N*K；在区间[128,2048]内的整数)：\n");
    scanf("%d %d %d",&M, &N, &K);
    printf("请输入线程数量(在区间[1,16]内的整数)：\n");
    scanf("%d",&thread_num);
    printf("请输入步长：\n");
    scanf("%d", &Inc);

    // 为矩阵A，B，C分配空间并将矩阵C初始化为全0矩阵
    A.resize(M*N);
    B.resize(N*K);
    C.assign(M*K, 0);
    
    // 初始化矩阵A和B并打印它们
    A = random_matrix_generator(M,N);
    B = random_matrix_generator(N,K);
    printMatrix(A,M,N);
    printMatrix(B,N,K);

    missions_per_thread = M / thread_num;   // 算出每个线程需要完成的任务数
    for(int i = 0; i < M; ++i){
        unassigned.insert(i);       // 初始化unassigned，此时还未分配任务，所以0～M-1均在unassigned中
    }

    FUNCTOR_ARGS** Args;
    Args = new FUNCTOR_ARGS*[thread_num];
    for(int i = 0; i < thread_num; ++i){
        Args[i] = new FUNCTOR_ARGS(M,N,K,missions_per_thread,Inc);  // 为每个线程分配任务
    }

    double using_time = parallel_for(0, M, Inc, functor, (void*)Args, thread_num);

    if(!unassigned.empty()){    // 此时对应M无法被线程数整除的情况，需要计算剩余的行与矩阵B的乘积，并加上这部分的用时
        auto start_time = chrono::high_resolution_clock::now();
        for(auto iter = unassigned.begin(); iter != unassigned.end(); iter++){
            int m = *iter;
            for(int k = 0; k < K; ++k){
                for(int n = 0; n < N; ++n){
                    C[m*K+k] += A[m*N+n] * B[n*K+k];
                }
            }
        }
        auto end_time = chrono::high_resolution_clock::now();
        using_time += (end_time-start_time).count()/1e9;
    }
    printf("Matrix C：\n");
    printMatrix(C, M, K);
    printf("在%d个线程并行的情况下计算大小为%d*%d的矩阵A和%d*%d的矩阵B的乘积所用的时间为：%lfs\n",thread_num,M,N,N,K,using_time);
    return 0;
}