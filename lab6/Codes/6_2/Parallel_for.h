#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

// 矩阵参数
#define M 500
#define N 500

// 矩阵w和u（u用于记录上一次迭代的结果）
extern double w[M][N];
extern double u[M][N];

// 参数：diff用于记录每次迭代后矩阵中最大的变化值；mean用于为初始的矩阵非边界部分赋值
extern double diff;
extern double mean;

enum LOCATIONS{HORIZONTAL, VERTICAL};   // 为矩阵边界赋值时边界的方位（行：水平；列：竖直）
enum FUNCTION_TYPES{ASSIGN_BORDER, ASSIGN_INTERNAL, SAVE_LAST_MAT, UPDATE_MAT, UPDATE_DIFF};    
//函数类型，用于在parallel_for中选择功能

void* assignBorder(void* args);     //给边界赋值并计算mean
void* assignInternal(void* args);   //给矩阵内部赋值为mean
void* saveLastMatrix(void* args);   //保存上一次迭代的矩阵信息
void* updateMatrix(void* args);     //根据热量扩散公式更新矩阵
void* updateDiff(void* args);   // 更新diff
void parallel_for(int type, int start,int end, int incr, void* (*functor_ptr)(void*), int thread_num );
// 使用pthread来模仿openmp的并行for循环分解、分配及执行机制
#endif