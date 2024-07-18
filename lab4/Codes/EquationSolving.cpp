#include<cstdio>
#include<pthread.h>
#include<cmath>
#include<chrono>
using namespace std;

double a, b, c;     // 方程的三个参数a,b,c
double square_B, A_times_C;     // 存储b*b的结果和a*c的结果
double delta;   // 存储delta的计算结果即b*b-4*a*c
bool square_B_is_OK, A_times_C_is_OK;   // 用来判断是否已经计算出中间结果b*b和a*c以及最终的delta

pthread_mutex_t mutex;      // 互斥锁
double x1,x2;   // 方程的两个根
bool solution_exist;    // 记录是否一元二次方程有解

// 计算b^2
void* calculate_square_B(void* id){
    square_B = b*b;
    square_B_is_OK = true;
    pthread_exit(NULL);
}

// 计算a*c
void* calculate_A_times_C(void* id){
    A_times_C = a*c;
    A_times_C_is_OK = true;
    pthread_exit(NULL);
}

// 计算delta
void* calculate_delta(void* id){
    // 此处要使用互斥锁，保证quare_B_is_OK和A_times_C_is_OK的访问和修改是安全的，这样才能保证delta的计算无误
    pthread_mutex_lock(&mutex);
    while(!square_B_is_OK || !A_times_C_is_OK){
        pthread_mutex_unlock(&mutex);
        pthread_mutex_lock(&mutex);
    }
    delta = square_B - 4*A_times_C;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

int main(){
    printf("请输入方程 a*x^2+b*x+c=0 的参数a、b、 c(a、b、c均为范围为[-100,100]的随机数，由于是一元二次方程，所以a不能为0)：\n");    
    scanf("%lf %lf %lf",&a,&b,&c);

    // 对标志量进行初始化
    square_B_is_OK = false;     
    A_times_C_is_OK = false;
    solution_exist = false;

    pthread_t handles[3];  // 为线程分配空间
    int thread_ids[3] = {0,1,2};     // 线程号
    pthread_mutex_init(&mutex, NULL);       // 初始化互斥锁
    auto start_time = chrono::high_resolution_clock::now();     // 开始计时

    // 创建并启动线程
    pthread_create(&handles[0], NULL, calculate_square_B, (void*)&thread_ids[0]);
    pthread_create(&handles[1], NULL, calculate_A_times_C, (void*)&thread_ids[1]);
    pthread_create(&handles[2], NULL, calculate_delta, (void*)&thread_ids[2]);

    for(int i = 0; i < 3; ++i){
        pthread_join(handles[i], NULL);     // 等待线程结束并回收线程资源
    }

    // delta大于或等于0，说明方程有解，计算方程的根
    if(delta >= 0){
        x1 = (-1*b - sqrt(delta)) / (2*a);
        x2 = (-1*b + sqrt(delta)) / (2*a);
        solution_exist = true;
    }
    auto end_time = chrono::high_resolution_clock::now();   // 结束计时
    auto using_time = (double)(end_time-start_time).count() ;   // 计算出时间，此处以纳秒为单位，若使用秒为单位，可能会因为精确度不够而显示为0
    
    // 方程有根，打印2个根
    if(solution_exist){     
        printf("方程的根为：\nx1 = %lf\nx2 = %lf\n", x1, x2);
    }

    // 方程无根
    else{
        printf("方程无解.\n");
    }
    printf("并行多线程求解方程%lfx^2+%lfx+%lf=0所用时间为：%lfns\n",a,b,c,using_time);

    pthread_mutex_destroy(&mutex);      // 销毁互斥锁

    // 下为串行方程求解
    start_time = chrono::high_resolution_clock::now();     // 开始计时
    delta = b*b - 4*a*c;
    if(delta>=0){
        x1 = (-1*b - sqrt(delta)) / (2*a);
        x2 = (-1*b + sqrt(delta)) / (2*a);
        solution_exist = true;
    }
    end_time = chrono::high_resolution_clock::now();   // 结束计时
    using_time = (double)(end_time-start_time).count() ;   // 计算出时间，此处以纳秒为单位，若使用秒为单位，可能会因为精确度不够而显示为0
    // 方程有根，打印2个根
    if(solution_exist){     
        printf("方程的根为：\nx1 = %lf\nx2 = %lf\n", x1, x2);
    }
    // 方程无根
    else{
        printf("方程无解.\n");
    }
    printf("串行求解方程%lfx^2+%lfx+%lf=0所用时间为：%lfns\n",a,b,c,using_time);
    return 0;
    
}