#include "Generator.hpp"
#include <iostream>
#include <ctime>
#include"mkl.h"
using namespace std;

// 计算通用串行矩阵乘法所用时间
double serial_time(vector<vector<double>> A, vector<vector<double>> B){
    clock_t start_time, end_time;   // start_time：记录乘法运算开始的时刻； end_time：记录乘法运算结束的时刻
    int M = A.size(), N = B.size(), K = B[0].size();    // 获取矩阵维度M，N，K
    vector<vector<double>> C(M, vector<double>(K,0));       // C=AB，C初始化为全0矩阵
    start_time = clock();
    for(int i = 0; i < M; ++i){
        for(int k = 0; k < K; ++k){
            for(int j = 0; j < N; ++j){
                C[i][k] += A[i][j] * B[j][k];       // 按照定义计算矩阵乘法的结果
            }
        }
    }
    end_time = clock();
    double using_time = (double)(end_time - start_time)/ CLOCKS_PER_SEC;     // 计算乘法运算所耗费的时间(s)
    printMatrix(C, M, K);
    return using_time;
}

// 计算调整循环顺序后的串行矩阵乘法运算的时间
double adjust_order_time(vector<vector<double>> A, vector<vector<double>> B){
    clock_t start_time, end_time;   // start_time：记录乘法运算开始的时刻； end_time：记录乘法运算结束的时刻
    int M = A.size(), N = B.size(), K = B[0].size();    // 获取矩阵维度M，N，K
    vector<vector<double>> C(M, vector<double>(K,0));       // C=AB，C初始化为全0矩阵
    start_time = clock();
    for(int i = 0; i  <  M; ++i){
        for(int j = 0; j < N; ++j){
            for(int k = 0; k < K;++k){
                C[i][k] += A[i][j] * B[j][k];
            }
        }
    }
    end_time = clock();
    double using_time = (double)(end_time - start_time)/ CLOCKS_PER_SEC;    // 计算乘法运算所耗费的时间(s)
    printMatrix(C, M, K);
    return using_time;
}

// 计算循环展开串行矩阵乘法所用时间
double loop_unrolling_time(vector<vector<double>> A, vector<vector<double>> B){
    clock_t start_time, end_time;   // start_time：记录乘法运算开始的时刻； end_time：记录乘法运算结束的时刻
    int M = A.size(), N = B.size(), K = B[0].size();    // 获取矩阵维度M，N，K
    vector<vector<double>> C(M, vector<double>(K,0));       // C=AB，C初始化为全0矩阵
    start_time = clock();
    for(int i = 0; i < M; i+=4){    //  对最外层循环进行展开，间隔为4
        for(int k = 0; k < K; ++k){    
            for(int j = 0; j < N; ++j){   
                C[i][k] += A[i][j] * B[j][k];       // 按照定义计算矩阵乘法的结果
                C[i+1][k] += A[i+1][j] * B[j][k];
                C[i+2][k] += A[i+2][j] * B[j][k];
                C[i+3][k] += A[i+3][j] * B[j][k];
            }
        }
    }
    end_time = clock();
    double using_time = (double)(end_time - start_time)/ CLOCKS_PER_SEC;     // 计算乘法运算所耗费的时间(s)
    printMatrix(C, M, K);
    return using_time;
}

// 使用Intel MKL库，使用dgemm进行矩阵计算
double MKL_cblas_dgemm_time(int M, int N, int K, double alpha, double beta){
    double *A, *B, *C;      // 3个矩阵

    // 分配对齐的内存缓存区
    A = (double*)mkl_malloc(M * N *sizeof(double), 64);
    B = (double*)mkl_malloc(N * K *sizeof(double), 64);
    C = (double*)mkl_malloc(M * K *sizeof(double), 64);
    
    if (A == NULL || B == NULL || C == NULL) {
     cout<<"\n ERROR: Can't allocate memory for matrices. Aborting... \n\n";
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }

    // 初始化随机数生成器
    random_device rd;   
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(-50.0,50.0);

    // 给矩阵A赋值并打印出矩阵A
    for(int i = 0; i < M * N; ++i){
        A[i] = distr(eng);
        cout<<A[i]<<" ";
    }
    cout<<endl;

    // 给矩阵B赋值并打印出矩阵B
    for(int i = 0; i < N * K; ++i){
        B[i] = distr(eng);
    }
    cout<<endl;

    // 矩阵C初始化为全0矩阵
    for(int i = 0; i < M * K; ++i){
        C[i] = 0;
    }
    
    clock_t start_time, end_time;
    start_time = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, alpha, A, N, B,K, beta, C, K);      // 调用cblas_dgemm，各参数含义见官网
    end_time = clock();

    // 打印输出矩阵C
    for(int i = 0; i < M * K; ++i){
        cout<<C[i]<<" ";
    }
    cout<<endl;
    
    // 释放分配的缓存
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    double using_time = (double)(end_time - start_time)/ CLOCKS_PER_SEC;     // 计算乘法运算所耗费的时间(s)
    return using_time;
}
int main(){
    int M, N, K;
    cout<<"请输入3个整数M,N,K(矩阵A的大小为M*N，矩阵B的大小为N*K；M，N，K的范围为[512, 2048]): "<<endl;
    cin >> M >> N >> K;
    int if_using_mkl = 0;
    cout<<"是否使用Intel MKL库？若是请输入1，否则请输入0：";
    cin>>if_using_mkl;
    if(if_using_mkl == 1){
        double alpha, beta;
        cout<<"接下来请输入两个比例因子alpha和beta，用于计算双精度矩阵的乘积：dgemm. 其计算公式为C = alpha*AB + beta*C"<<endl;
        cout<<"请输入矩阵A和B乘积的比例因子alpha：";
        cin>>alpha;
        cout<<"请输入矩阵C的比例因子beta：";
        cin>>beta;
        double MKL_time = MKL_cblas_dgemm_time(M,N,K,alpha,beta);
        cout<<"调用Intel MKL库进行矩阵乘法运算所用时间为："<<MKL_time<<" s"<<endl;
    }
    else{
        vector<vector<double>> A, B;
    // 随机生成矩阵A、B
        A = random_matrix_generator(M,N);
        B = random_matrix_generator(N,K);
        printMatrix(A, M, N);
        printMatrix(B, N, K);
    //   double STime = serial_time(A,B);    // STime：使用通用串行矩阵乘法计算AB的时间
    //    double ATime = adjust_order_time(A,B);      // ATime：使用调整循环顺序的串行矩阵乘法计算AB的时间
        double LUTime = loop_unrolling_time(A,B);
    // cout<<"使用通用串行矩阵乘法所用的时间为："<<STime<<" s"<<endl;
    //     cout<<"使用调整循环顺序后的串行矩阵乘法所用的时间为："<<ATime<<" s"<<endl;
    //   cout<<"进行编译优化后使用通用串行矩阵乘法所用的时间为："<<STime<<" s"<<endl;
        cout<<"进行循环展开后使用通用串行矩阵乘法所用的时间为："<<LUTime<<" s"<<endl;
    }
    return 0;
}