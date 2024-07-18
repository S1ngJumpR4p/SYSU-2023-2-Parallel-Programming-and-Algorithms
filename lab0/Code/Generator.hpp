#ifndef GENERATOR_HPP
#define GENERATOR_HPP

#include<iostream>
#include<random>
#include<vector>
using namespace std;

// 随机生成一个 row s* cols 的双精度浮点数矩阵
vector<vector<double>>  random_matrix_generator(int rows, int cols){
    vector<vector<double>> matrix(rows, vector<double>(cols,0));      
    // 初始化随机数生成器
    random_device rd;   
    default_random_engine eng(rd());
    uniform_real_distribution<double> distr(-50.0,50.0);

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            matrix[i][j] = distr(eng);
        }
    }
    return matrix;
}

// 打印输出矩阵
void printMatrix(vector<vector<double>> M, int rows, int cols){
    for(int i = 0;i < rows;++i){
        for(int j = 0; j < cols; ++j){
            cout << M[i][j] <<" ";
        }
        cout<<endl;
    }
}
#endif