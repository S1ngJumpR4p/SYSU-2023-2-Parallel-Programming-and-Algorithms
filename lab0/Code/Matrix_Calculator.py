# coding:utf-8 
import Generator
import time

def serial_time(A, B):
    M, N, K = len(A), len(B), len(B[0]) # 获取矩阵维度M，N，K
    C = [[0.0 for _ in range(K)] for _ in range(M)]   # 初始化结果矩阵C（M*K矩阵）
    start_time = time.time()    # 记录矩阵乘法开始时刻
    for i in range(M):
        for k in range(K):
            for j in range(N):
                C[i][k] = C[i][k] + A[i][j] * B[j][k]
    end_time = time.time()  # 记录矩阵乘法结束时刻
    Generator.printMatrix(C)    # 打印输出矩阵C
    return end_time-start_time     # 返回矩阵乘法所用时间


print("请输入3个正整数M、N、K（矩阵A的大小为M * K，矩阵B的大小为N * K；M、N、K的范围为[512,2048]）")
M = int(input("请输入M："))
N = int(input("请输入N："))
K = int(input("请输入K："))

# 生成矩阵A并打印输出A
A = Generator.random_matrix_generator(M,N)
Generator.printMatrix(A)

# 生成矩阵B并打印输出B
B = Generator.random_matrix_generator(N,K)
Generator.printMatrix(B)

STime = serial_time(A, B)   # 使用通用串行矩阵乘法计算AB所用的时间
#  ATime = adjust_order_time(A, B)     # 使用调整循环顺序后的串行矩阵乘法计算AB所用的时间
print("使用python实现串行矩阵乘法所用时间为："+str(STime)+" s")
