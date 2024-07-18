#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#include<vector>
#include<cstdio>
#include<set>
#include <chrono>
using namespace std;
extern int M,N,K,thread_num,Inc, missions_per_thread;   // 矩阵规模参数、线程数、步长、每个线程要完成的任务数量
extern set<int> unassigned;     // 记录还未被分配到线程的任务编号
extern vector<double> A, B, C;      // 矩阵A，B，C
void printMatrix(vector<double>& mat, int row, int col);    //按照row*col的规格 打印矩阵
void* functor(void* args);      // 执行每个线程被分配到的任务
double parallel_for(int start, int end, int inc, void* (*functor_ptr)(void*), void*args, int thread_num);    
// 模仿OpenMP的omp_parallel_for构造基于Pthreads的并行for循环分解、分配及执行

// 定义一个结构体用来存放每个线程执行任务所需的参数
struct  FUNCTOR_ARGS
{
    /* 
        M、N、K：矩阵规模参数
        missions_per_thread：该线程需完成的任务数量
        Incr：步长
        Indices：每个线程要完成的任务的编号
        构造函数：对M, N, K, missions_per_thread, Incr进行赋值，然后先从unassigned里找到当前最小的任务编号I，将其
        作为该线程要完成的第一个任务并记录其为current_index，然后看看current_index+Incr是否会越界，若不越界则将
        current_index+Incr放入Indices中并将current_index更新为current_index+Incr，否则取unassigned中最小的任务
        编号放入Indices中，将其赋值给current_index，继续迭代直到已经领取了missions_per_thread个任务。注意，在
        从unassigned中取任务到Indices中时要移除unassigned中对应的任务编号。
     */
    int M, N, K, missions_per_thread, Incr;      
    vector<int> Indices;
    FUNCTOR_ARGS(int m,int n, int k, int missions, int inc){
        this->M = m;
        this->N = n;
        this->K = k;
        this->missions_per_thread = missions;
        this->Incr = inc;
        int current_index = *unassigned.begin();    // 刚领取到的任务编号
        this->Indices.emplace_back(current_index);
        unassigned.erase(current_index);
        int cnt = 1;    // 当前已经领取到的任务数量
        bool flag;      // 当前任务编号加上Incr后是否会越界
        while(cnt < this->missions_per_thread){
            flag = current_index + this->Incr < this->M;
            if(flag){
                current_index += this->Incr;
                this->Indices.emplace_back(current_index);
                unassigned.erase(current_index);
            }
            else{
                current_index = *unassigned.begin();
                this->Indices.emplace_back(current_index);
                unassigned.erase(current_index);
            }
            cnt++;
        }
    }
};

#endif