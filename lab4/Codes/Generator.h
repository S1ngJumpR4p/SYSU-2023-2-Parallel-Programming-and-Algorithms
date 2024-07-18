#ifndef GENERATOR_H
#define GENERATOR_H
#include<random>
#include<utility>
using namespace std;

// 生成指定范围内的点
pair<double,double> random_point_generator(){
    // 初始化随机数生成器
    random_device rd;   
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(0, 1);
    return make_pair(distr(eng),distr(eng));
}


// 生成指定大小的归一化后的点集
vector< pair<double,double> > random_points_set_generator(int num){
    vector< pair<double,double> > points;
    points.resize(num);
    for(auto& point : points){
        point = random_point_generator();
    }
    return points;
}
#endif