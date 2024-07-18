#ifndef GENERATOR_H
#define GENERATOR_H

#include<vector>
#include<random>
using namespace std;
// 随机生成一个指定大小的矩阵，存储于一个一维数组
vector<double> random_matrix_generator(int rows, int cols){
    vector<double> mat(rows*cols);

    // 初始化随机数生成器
    random_device rd;   
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(-50.0,50.0);

    // 赋值
    for(int i = 0; i < rows*cols; ++i){
        mat[i] = distr(eng);
    }

    return mat;
}

#endif