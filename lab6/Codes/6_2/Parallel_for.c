#include "Parallel_for.h"
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
pthread_mutex_t Mutex;
// 执行边界赋值时所需参数
struct BORDER_ARGS{
    /*
        start：起始索引
        end：终止索引（取不到）
        increment：步长
        location：方位，是给行边界赋值还是给列边界赋值
    */
    int start, end, increment;
    enum LOCATIONS location;
};

// 为矩阵内部赋值时所需参数
struct INTERNAL_ARGS{
    /*
        start：起始行号
        end：终止行号（取不到）
        increment：步长
        value：要赋的值（mean）
    */
   int start, end, increment;
   double value;
};

// 保存上一次迭代的矩阵信息所需参数
struct SAVE_LAST_MAT_ARGS{
    /*
        start：起始行号
        end：终止行号（取不到）
        increment：步长
    */
   int start, end, increment;
};

// 更新矩阵内部值时所需参数
struct UPDATE_MAT_ARGS{
    /*
        start：起始行号
        end：终止行号（取不到）
        increment：步长
    */
   int start, end, increment;
};

// 更新每次迭代的矩阵最大变化值所需参数
struct  UPDATE_DIFF_ARGS
{
     /*
        start：起始行号
        end：终止行号（取不到）
        increment：步长
        maxdiff：每个线程对应位置的矩阵元素最大变化值
    */
   int start, end, increment;
   double maxdiff;
};

void* assignBorder(void* args){
    struct BORDER_ARGS* Args = (struct BORDER_ARGS*)args;   // 进行类型转换，用于获取参数
    if(Args->location == HORIZONTAL){   // 要对顶上和地下两行赋值
        for(int i = Args->start; i < Args->end; i+=Args->increment){
            w[0][i] = 0;
            w[M-1][i] = 100;
            pthread_mutex_lock(&Mutex);
            mean += w[0][i] + w[M-1][i];    // 使用互斥锁保护mean的计算
            pthread_mutex_unlock(&Mutex);
        }
    }
    else{   // 要对最左列和最右列两列赋值，注意不包含四个顶点
        for(int i = Args->start; i < Args->end; i+=Args->increment){
            w[i][0] = 100;
            w[i][N-1] = 100;
            pthread_mutex_lock(&Mutex);
            mean += w[i][0] + w[i][N-1];    // 使用互斥锁保护mean的计算
            pthread_mutex_unlock(&Mutex);
        }
    }
    pthread_exit(NULL);     // 退出线程
}

void* assignInternal(void* args){
    struct INTERNAL_ARGS* Args = (struct INTERNAL_ARGS*)args;   // 进行类型转换，用于获取参数
    for(int i = Args->start; i < Args->end; i+=Args->increment){
        for(int j = 1; j < N-1; ++j){
            w[i][j] = Args->value;      // 矩阵内部赋值
        }
    }
    pthread_exit(NULL);     // 退出线程
}

void* saveLastMatrix(void* args){
    struct SAVE_LAST_MAT_ARGS* Args = (struct SAVE_LAST_MAT_ARGS*)args;     // 进行类型转换，用于获取参数
    for(int i = Args->start; i < Args->end; i+=Args->increment){
        for(int j = 0; j < N; ++j){
            u[i][j] = w[i][j];      // 保存上一次迭代的结果
        }
    }
    pthread_exit(NULL);     // 退出线程
}

void* updateMatrix(void* args){
    struct UPDATE_MAT_ARGS* Args = (struct UPDATE_MAT_ARGS*)args;    // 进行类型转换，用于获取参数
    for(int i = Args->start; i < Args->end; i+=Args->increment){
        for(int j = 1; j < N-1; ++j){
            w[i][j] = (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]) / 4;  // 根据公式更新矩阵内部的值
        }
    }
    pthread_exit(NULL);     // 退出线程
}

void* updateDiff(void* args){
    struct UPDATE_DIFF_ARGS* Args = (struct UPDATE_DIFF_ARGS*)args;     // 进行类型转换，用于获取参数
    for(int i = Args->start; i < Args->end; i+=Args->increment){
        for(int j = 1; j < N-1; ++j){
            if(Args->maxdiff < fabs(w[i][j]-u[i][j])){
                Args->maxdiff = fabs(w[i][j]-u[i][j]);  // 更新矩阵内部最大的变化值
            }
        }
    }
    pthread_mutex_lock(&Mutex);
    if(diff < Args->maxdiff){
                diff = Args->maxdiff;
            }
    pthread_mutex_unlock(&Mutex);
    pthread_exit(NULL);     // 退出线程
}

void parallel_for(int type, int start,int end, int incr, void* (*functor_ptr)(void*), int thread_num ){
    pthread_mutex_init(&Mutex,NULL);    // 初始化互斥锁
    if(type == ASSIGN_BORDER){
        pthread_t* handles = (pthread_t*)malloc(sizeof(pthread_t) * thread_num * 2);    // 为线程分配空间
        // 分成2*thread_num个线程是因为行边界需要thread_num个线程，列边界需要thread_num个线程
        struct BORDER_ARGS** Args = (struct BORDER_ARGS**)malloc(sizeof(struct BORDER_ARGS*) * thread_num * 2);
        int rowChunksize = N / thread_num, colChunksize = (M-2) / thread_num;
        for(int i = 0; i < thread_num; ++i){
            // 初始化参数
            Args[i] = (struct BORDER_ARGS*)malloc(sizeof(struct BORDER_ARGS));
            Args[i+thread_num] = (struct BORDER_ARGS*)malloc(sizeof(struct BORDER_ARGS));
            Args[i]->location = HORIZONTAL;
            Args[i]->increment = incr;
            Args[i+thread_num]->location = VERTICAL;
            Args[i+thread_num]->increment = incr;
            if(i == 0){     // 第一个线程要把起点确定好，注意行和列的区别
                Args[0]->start = 0;
                Args[thread_num]->start = 1;
                Args[0]->end = rowChunksize;
                Args[thread_num]->end = colChunksize+1;
            }
            else if(i == thread_num-1){     // 最后一个线程要把剩余的所有任务都完成
                Args[thread_num-1]->start = (thread_num-1)*rowChunksize;
                Args[2*thread_num-1]->start = (thread_num-1)*colChunksize+1;
                Args[thread_num-1]->end = N;
                Args[2*thread_num-1]->end = M-1;
            }
            else{   // 一般情况
                Args[i]->start = i*rowChunksize;
                Args[i]->end = Args[i]->start + rowChunksize;
                Args[i+thread_num]->start = i*colChunksize + 1;
                Args[i+thread_num]->end = Args[i+thread_num]->start + colChunksize;
            }

            // 创建并启动线程
            pthread_create(&handles[i],NULL,functor_ptr,Args[i]);
            pthread_create(&handles[i+thread_num],NULL,functor_ptr,Args[i+thread_num]);
        }
        for(int i = 0; i < thread_num; ++i){
            // 回收资源
            pthread_join(handles[i],NULL);
            pthread_join(handles[i+thread_num],NULL);
            free(Args[i]);
            free(Args[i+thread_num]);
        }
        free(Args);
    }
    else if(type == ASSIGN_INTERNAL){
        pthread_t* handles = (pthread_t*)malloc(sizeof(pthread_t) * thread_num);    // 为线程分配空间
        struct INTERNAL_ARGS** Args = (struct INTERNAL_ARGS**)malloc(sizeof(struct INTERNAL_ARGS*)*thread_num);
        int Chunksize = (M-2)/thread_num;
        for(int i = 0; i < thread_num; ++i){
            // 初始化参数
            Args[i] = (struct INTERNAL_ARGS*)malloc(sizeof(struct INTERNAL_ARGS));
            Args[i]->start = start + i*Chunksize;
            Args[i]->increment = incr;
            Args[i]->value = mean;
            if(i == thread_num-1){
                Args[i]->end = end;
            }
            else{
                Args[i]->end = Args[i]->start + Chunksize;
            }
            pthread_create(&handles[i],NULL,functor_ptr,Args[i]);       // 创建并启动线程
        }
        for(int i = 0; i < thread_num; ++i){
            // 回收资源
            pthread_join(handles[i],NULL);
            free(Args[i]);
        }
       
        free(Args);
    }
    else if(type == SAVE_LAST_MAT){
        pthread_t* handles = (pthread_t*)malloc(sizeof(pthread_t) * thread_num);    // 为线程分配空间
        struct SAVE_LAST_MAT_ARGS** Args = (struct SAVE_LAST_MAT_ARGS**)malloc(sizeof(struct SAVE_LAST_MAT_ARGS*)*thread_num);
        int Chunksize = M/thread_num;
        for(int i = 0; i < thread_num; ++i){
            // 初始化参数
            Args[i] = (struct SAVE_LAST_MAT_ARGS*)malloc(sizeof(struct SAVE_LAST_MAT_ARGS));
            Args[i]->start = start + i*Chunksize;
            Args[i]->increment = incr;
            if(i == thread_num-1){
                Args[i]->end = end;
            }
            else{
                Args[i]->end = Args[i]->start + Chunksize;
            }
            pthread_create(&handles[i],NULL,functor_ptr,Args[i]);       // 创建并启动线程
        }
        for(int i = 0; i < thread_num; ++i){
            // 回收资源
            pthread_join(handles[i],NULL);
            free(Args[i]);
        }
        free(Args);
    }
    else if(type == UPDATE_MAT){
        pthread_t* handles = (pthread_t*)malloc(sizeof(pthread_t) * thread_num);    // 为线程分配空间
        struct UPDATE_MAT_ARGS** Args = (struct UPDATE_MAT_ARGS**)malloc(sizeof(struct UPDATE_MAT_ARGS*)*thread_num);
        int Chunksize = (M-2)/thread_num;
        for(int i = 0; i < thread_num; ++i){
            // 初始化参数
            Args[i] = (struct UPDATE_MAT_ARGS*)malloc(sizeof(struct UPDATE_MAT_ARGS));
            Args[i]->start = start + i*Chunksize;
            Args[i]->increment = incr;
            if(i == thread_num-1){
                Args[i]->end = end;
            }
            else{
                Args[i]->end = Args[i]->start + Chunksize;
            }
            pthread_create(&handles[i],NULL,functor_ptr,Args[i]);       // 创建并启动线程
        }
        for(int i = 0; i < thread_num; ++i){
            // 回收资源
            pthread_join(handles[i],NULL);
            free(Args[i]);
        }
        free(Args);
    }
    else{
        pthread_t* handles = (pthread_t*)malloc(sizeof(pthread_t) * thread_num);    // 为线程分配空间
        struct UPDATE_DIFF_ARGS** Args = (struct UPDATE_DIFF_ARGS**)malloc(sizeof(struct UPDATE_DIFF_ARGS*)*thread_num);
        int Chunksize = (M-2)/thread_num;
        for(int i = 0; i < thread_num; ++i){
            // 初始化参数
            Args[i] = (struct UPDATE_DIFF_ARGS*)malloc(sizeof(struct UPDATE_DIFF_ARGS));
            Args[i]->start = start + i*Chunksize;
            Args[i]->increment = incr;
            Args[i]->maxdiff = 0.0;
            if(i == thread_num-1){
                Args[i]->end = end;
            }
            else{
                Args[i]->end = Args[i]->start + Chunksize;
            }
            pthread_create(&handles[i],NULL,functor_ptr,Args[i]);       // 创建并启动线程
        }
        for(int i = 0; i < thread_num; ++i){
            // 回收资源
            pthread_join(handles[i],NULL);
            free(Args[i]);
        }
        free(Args);
    }
    pthread_mutex_destroy(&Mutex);
}