# coding:utf-8 
import random

# 随机生成大小为rows * cols的双精度浮点数矩阵 
def random_matrix_generator(rows, cols):
    matrix = []     
    for _ in range(rows):
        row = []    # 行
        for _ in range(cols):
            row.append(random.random() * 10 - 5)       # random.random()函数返回一个[0,1]的浮点数x，x * 10 - 5 代表[-5,5]的浮点数
        matrix.append(row)
    return matrix

# 打印输出指定矩阵
def printMatrix(A):
    for i in range(len(A)):
        if i == 0:
            print("[")
        print(A[i])
        if i == len(A)-1:
            print("]")
        else:
            print("\n")
