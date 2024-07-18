#include"Generator.h"
#include<chrono>
#include<pthread.h>
#include<cstdio>
#include<vector>

int point_num;      // 总点数
int thread_num;     // 线程数
int in_circle_point_num = 0;    // 在内切圆内的点数
int points_per_thread;      // 每个线程要统计的点数
const pair<double,double> origin = make_pair(0.0,0.0);      // 固定原点
vector< pair<double,double> > points;     // 点集
vector<int> in_circle_points_per_thread;    // 各个线程统计后在内切圆内的点数
double PI;

// 线程内统计在内切圆内的点数
void* Count_Points(void* id){
    int thread_id = *((int*)id);      // 获取线程号
    int begin_index = thread_id * points_per_thread, end_index = (thread_id+1) * points_per_thread;  
    // 统计的点的索引范围
    
    // 遍历要统计的所有点
    for(int i = begin_index; i < end_index; ++i){
        double x_distance = points[i].first - origin.first, y_distance = points[i].second - origin.second;
        double square_distance = x_distance*x_distance + y_distance * y_distance;
        // 计算当前点与原点的距离
        
        if(square_distance <= 1){  // 在内切圆内, 圆周上也算进去，小于等于1是因为已经进行了归一化
            in_circle_points_per_thread[thread_id] += 1;
        }
    }
    pthread_exit(NULL);
}

int main(){

    printf("请输入总点数和线程数：\n");
    scanf("%d %d",&point_num, &thread_num);
    
    const int THREAD_NUM = thread_num;
    points_per_thread = point_num / THREAD_NUM;    // 算出每个线程要统计多少个点  
    in_circle_points_per_thread.assign(THREAD_NUM+1, 0);      // 初始化各个线程内在内切圆内的点数
    points = random_points_set_generator(point_num);   // 生成在正方形内的随机点集

    pthread_t handles[THREAD_NUM];  // 为线程分配空间
    int thread_ids[THREAD_NUM];     // 线程号

    auto start_time = chrono::high_resolution_clock::now();     // 记录开始时间

    for(int i = 0; i < THREAD_NUM; ++i){
        thread_ids[i] = i;  // 线程号
        pthread_create(&handles[i], NULL, Count_Points, (void*)&thread_ids[i]);     // 创建并启动线程
    }

    for(int i = 0; i < THREAD_NUM; ++i){
        pthread_join(handles[i],NULL);   // 等待线程结束并回收线程资源
    }

    if(point_num % THREAD_NUM != 0){        // 点数不一定能被线程数整除，应统计剩余的(point_num%THREAD_NUM)个点
        int cnt = 0;
        for(int i = THREAD_NUM * points_per_thread; i < point_num; ++i){
            double x_distance = points[i].first - origin.first, y_distance = points[i].second - origin.second;
            double square_distance = x_distance*x_distance + y_distance * y_distance;
            // 计算当前点与原点的距离
        
            if(square_distance <= 1){  // 在内切圆内, 圆周上也算进去
                cnt += 1;
            }
        }
        in_circle_points_per_thread[THREAD_NUM] = cnt;
    }

    // 将各个线程统计到的在内切圆内的点数加起来
    for(auto& num : in_circle_points_per_thread){
        in_circle_point_num += num;
    }
    PI = (double)in_circle_point_num * 4 / (double)point_num;   // 计算π值

    auto end_time = chrono::high_resolution_clock::now();   // 记录结束时间
    auto using_time = (double)(end_time - start_time).count() / 1e9;  

    printf("\n总点数为：%d\n落在内切圆内的点数为：%d\n估算的π值为：%lf\n线程数为：%d\n所用时间为：%lfs\n",point_num,in_circle_point_num,PI,THREAD_NUM,using_time);
    return 0;
}