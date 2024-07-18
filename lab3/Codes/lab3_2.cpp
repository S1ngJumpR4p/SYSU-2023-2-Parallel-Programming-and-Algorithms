#include<cstdio>
#include<iostream>  // 接收string字符串
#include<cstdlib>
#include<chrono>    // 用来计时，不用clock()是因为其计算的是处理器的时间，不是实际时间
#include<vector>
#include<cmath>     // 需要使用幂运算
#include<random>    // 用于生成随机数
#include<pthread.h>     // 使用pthread库
#include<string>    // 用于截取子串
using namespace std;

#define MEGA (long long)(pow(2, 20))    // 定义1M(2^20)

vector<double> Array;   // 要求和的数组
double global_sum = 0.0;    // 全局数组总和
pthread_mutex_t Mutex;      // 互斥锁
long long nums_per_thread;      // 每个线程要求和的数的数量

// 随机生成一个指定大小的数组
vector<double> random_array_generator(long long num){
    size_t length = (size_t)num;
    vector<double> res(length);

     // 初始化随机数生成器
    random_device rd;   
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(0,1);

    for(size_t i = 0; i < length; ++i){
        res[i] = distr(eng);
    }
    return res;
}

// 求和线程
void* calculate_sum(void* id){
    int thread_id = *((int*)id);    // 获取线程号
    long long begin = thread_id * nums_per_thread, end = (thread_id+1) * nums_per_thread;   // 求和起点和终点：[begin,end)
    double local_sum = 0.0;     // 局部和
    for(long long i = begin; i < end; ++i){
        local_sum += Array[i];
    }
    // 使用互斥锁保证将局部和加到全局和
    pthread_mutex_lock(&Mutex);     // 上锁
    global_sum += local_sum;    
    pthread_mutex_unlock(&Mutex);   // 解锁
    pthread_exit(NULL);     // 终止线程
}
int main(){
    string len;   // 数组长度(未乘1M)
    bool flag = false;  //判断是否有输入"M"，若有"M"，则要进行处理乘以1M(2^20)
    int num;    // 线程数量
    printf("请输入数组长度(数组长度为输入范围在[1M,128M]的整数可以输入具体的整数也可以输入如\"1M、1.25M\"的数)：\n");
    cin>>len;
    printf("请输入线程数量(线程数量为[1,16]的整数)：\n");
    scanf("%d", &num);
    const int THREAD_NUM = num; 

    // 有"M" 
    if(len[len.length()-1] == 'M'){
        len.substr(0, len.length()-2);  // 提取前面的数字部分
        flag = true;    // 将flag改为true，后面需要再乘以1M(2^20)
    }
   double Len = stod(len);   // 将数字部分转换为数字
    if(flag){
        Len *= MEGA;
    }   
    long long Length = (long long)Len;
    nums_per_thread = Length / THREAD_NUM;  // 计算出每个线程的要求和的数的数量
    Array = random_array_generator(Length);     // 随机生成指定大小的数组

    pthread_t handles[THREAD_NUM];  // 为线程分配空间
    int pthread_ids[THREAD_NUM];    // 线程号集合
    pthread_mutex_init(&Mutex, NULL);   // 初始化互斥锁
    auto start_time = chrono::high_resolution_clock::now();     // 记录开始时间
    for(int i = 0; i < THREAD_NUM; ++i){
        pthread_ids[i] = i;
        pthread_create(&handles[i], NULL, calculate_sum, (void*)&pthread_ids[i]);   // 创建并启动线程
    }

    for(int i = 0; i < THREAD_NUM; ++i){
        pthread_join(handles[i], NULL);     // 等待线程结束并回收线程资源
    }

    if(Length % THREAD_NUM != 0){
         // 可能数组长度未必能被THREAD_NUM整除，因此要将对剩余的(Length % THREAD_NUM)进行秋和
        for(size_t i = THREAD_NUM * nums_per_thread; i < Array.size(); ++i){
            global_sum += Array[i];
        }
    }
    auto end_time = chrono::high_resolution_clock::now();     // 记录开始时间
    auto using_time = (double)(end_time - start_time).count() / 1e9;  
    /*
        使用chrono::high_resolution_clock::now()算出来的时间差(chrono::high_resolution_clock::duration<double>)是纳
        秒级别的，所以要除以1e9也即10的9次方才能转换成秒
    */
    printf("数组如下：\n");
    for(size_t i = 0; i < Array.size();++i){
        printf("%lf ",Array[i]);
    }
    printf("\n");
    printf("在%d个线程并行的情况下对大小为%s的数组求和的结果为%lf，所用时间为：%lf s.\n",THREAD_NUM,len.c_str(), global_sum, using_time);
    pthread_mutex_destroy(&Mutex);   // 销毁锁
    return 0;
}