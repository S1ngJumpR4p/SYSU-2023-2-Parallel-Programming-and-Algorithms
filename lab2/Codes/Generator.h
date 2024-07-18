/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2024-04-01 19:23:25
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-04-03 10:22:35
 * @FilePath: /codes/parallel/lab2/Generator.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
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
    uniform_real_distribution<double> distr(-5,5);

    // 赋值
    for(int i = 0; i < rows*cols; ++i){
        mat[i] = distr(eng);
    }

    return mat;
}

#endif