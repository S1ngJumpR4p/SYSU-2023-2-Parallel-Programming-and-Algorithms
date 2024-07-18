#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <mpi.h>
using namespace std;

int main();
void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
double cpu_time(void);
double ggl(double *ds);
void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn);
void timestamp();

int myrank, process_num;    // myrank：进程号；process_num：进程数
 
int main(){
    double ctime, ctime1, ctime2, error, flops, fnm1, mflops, sgn, z0, z1;
    int first, i, icase, it, ln2, n, nits = 10000;
    static double seed;
    double *w, *x, *y, *z;

    // 初始化MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_num);
    if( myrank ==0 ){
        timestamp();
        cout << "\n"
            << "FFT_SERIAL\n"
            << "  C++ version\n"
            << "\n"
            << "  Demonstrate an implementation of the Fast Fourier Transform\n"
            << "  of a complex data vector.\n";
        cout << "\n";
        cout << "  Accuracy check:\n";
        cout << "\n";
        cout << "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n"
            << "\n"
            << "             N      NITS    Error         Time          Time/Call     MFLOPS\n"
            << "\n";
    }

    seed = 331.0;
    n = 1;

    for (ln2 = 1; ln2 <= 20; ln2++)
    {
        n = 2 * n;

        w = new double[n];
        x = new double[2 * n];
        y = new double[2 * n];
        z = new double[2 * n];

        first = 1;

        for (icase = 0; icase < 2; icase++)
        {
            if( myrank == 0 ){      // 在0号进程初始化数组x和z
                if (first){
                    for (i = 0; i < 2 * n; i = i + 2){
                        z0 = ggl(&seed);
                        z1 = ggl(&seed);
                        x[i] = z0;
                        z[i] = z0;
                        x[i + 1] = z1;
                        z[i + 1] = z1;
                    }
                }
                else{
                    for (i = 0; i < 2 * n; i = i + 2){
                        z0 = 0.0;
                        z1 = 0.0;
                        x[i] = z0;
                        z[i] = z0;
                        x[i + 1] = z1;
                        z[i + 1] = z1;
                    }
                }
            }
            cffti(n, w);
            MPI_Barrier(MPI_COMM_WORLD);    // 阻塞，等到所有进程都计算出所需的三角函数值
            if (first)
            {
                sgn = +1.0;
                cfft2(n, x, y, w, sgn); // 先进行DFT
                MPI_Barrier(MPI_COMM_WORLD);    // 阻塞，保证所有进程都完成计算
                sgn = -1.0;
                cfft2(n, y, x, w, sgn); // 再进行IDFT
                MPI_Barrier(MPI_COMM_WORLD);    // 阻塞，保证所有进程都完成计算
                
                if( myrank == 0){       // 因为最后的计算结果都会汇总到0号进程，因此在0号进程计算总误差即可
                    fnm1 = 1.0 / (double)n;
                    error = 0.0;
                    for (i = 0; i < 2 * n; i = i + 2){
                        error = error + pow(z[i] - fnm1 * x[i], 2) + pow(z[i + 1] - fnm1 * x[i + 1], 2);
                    }
                    error = sqrt(fnm1 * error);
                    cout << "  " << setw(12) << n << "  " << setw(8) << nits << "  " << setw(12) << error;
                }

                first = 0;
            }
            else
            {
                if( myrank == 0 ){  // 前面的误差汇总到0号进程，要与前面输出的信息相对应，因此在0号进程进行计时
                    ctime1 = cpu_time();
                }

                for (it = 0; it < nits; it++)
                {
                    sgn = +1.0;
                    cfft2(n, x, y, w, sgn);
                    sgn = -1.0;
                    cfft2(n, y, x, w, sgn);
                }
                if( myrank == 0 ){      // 仅在0号进程计时
                    ctime2 = cpu_time();
                    ctime = ctime2 - ctime1;

                    flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);

                    mflops = flops / 1.0E+06 / ctime;

                    cout << "  " << setw(12) << ctime << "  " << setw(12) << ctime / (double)(2 * nits) << "  " << setw(12) << mflops << "\n";
                }
            }
        }
        if ((ln2 % 4) == 0){
            nits = nits / 10;
        }
        if (nits < 1){
            nits = 1;
        }
        delete[] w;
        delete[] x;
        delete[] y;
        delete[] z;
    }
    if( myrank == 0 ){
        cout << "\n";
        cout << "FFT MPI PARALLEL(UNPACKED):\n";
        cout << "  Normal end of execution.\n";
        cout << "\n";
        timestamp();
    }
    MPI_Finalize();
    return 0;
}

void ccopy(int n, double x[], double y[]) {
    int block = n / process_num;    // 分块大小

    // 对每个进程的数组块进行复制
    for( int i = myrank * block; i < (myrank+1) * block; ++i){
        y[i * 2 + 0] = x[i * 2 + 0];
        y[i * 2 + 1] = x[i * 2 + 1];
    }

    // 接下来进行进程间的通信
    if( myrank ){   // 非0进程，将复制结果发给0号进程的y的对应位置
        MPI_Send(&y[myrank * block * 2], 2 * block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else{   // 0号进程，在对应位置接收来自其他进程发送过来的复制结果
        for(int i = 1; i < process_num; ++i){
            MPI_Recv(&y[i * block * 2], 2 * block, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    return;
}

void cfft2(int n, double x[], double y[], double w[], double sgn)
{
    int j, m, mj, tgle;

    m = (int)(log((double)n) / log(1.99));
    mj = 1;

    tgle = 1;

    // 使用MPI_Bcast将数组x和y广播出去
    MPI_Bcast(x, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

    if (n == 2){
        return;
    }

    for (j = 0; j < m - 2; j++)
    {
        mj = mj * 2;
        if (tgle){
            // 由于前面已经使用step函数进行了蝶形运算，数组x和y是有更新的，为了确保运算的准确性，在每次进行蝶形运算之前都要使用MPI_Bcast广播更新后的数组x和y
            MPI_Bcast(x, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(y, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);            
            step(n, mj, &y[0 * 2 + 0], &y[(n / 2) * 2 + 0], &x[0 * 2 + 0], &x[mj * 2 + 0], w, sgn);
            tgle = 0;
        }
        else{
            // 由于前面已经使用step函数进行了蝶形运算，数组x和y是有更新的，为了确保运算的准确性，在每次进行蝶形运算之前都要使用MPI_Bcast广播更新后的数组x和y
            MPI_Bcast(x, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(y, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);                  
            step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
            tgle = 1;
        }
    }
    if (tgle){
            // 由于前面已经使用step函数进行了蝶形运算，数组x和y是有更新的，为了确保能够正确拷贝数组，因此需要使用MPI_Bcast广播更新后的数组x和y
            MPI_Bcast(x, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(y, 2 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);            
            ccopy(n, y, x);
    }

    mj = n / 2;
    // 由于前面已经使用ccopy函数进行了数组拷贝，数组x和y是有更新的，为了确保运算的准确性，在每次进行蝶形运算之前都要使用MPI_Bcast广播更新后的数组x和y
    MPI_Bcast(x, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);        
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

    return;
}

void cffti(int n, double w[]) // 使用欧拉公式计算出相应的cos和sin值
{
    double arg, aw;
    int i, n2;
    const double pi = 3.141592653589793;

    n2 = n / 2;
    aw = 2.0 * pi / ((double)n);

    for (i = 0; i < n2; i++)
    {
        arg = aw * ((double)i);
        w[i * 2 + 0] = cos(arg);
        w[i * 2 + 1] = sin(arg);
    }
    return;
}

double cpu_time(void) // 计算处理器时间
{
    double value;

    value = (double)clock() / (double)CLOCKS_PER_SEC;

    return value;
}

double ggl(double *seed) // 生成随机数
{
    double d2 = 0.2147483647e10;
    double t;
    double value;

    t = *seed;
    t = fmod(16807.0 * t, d2);
    *seed = t;
    value = (t - 1.0) / (d2 - 1.0);

    return value;
}

void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn){
    double ambr, ambu;
    int j, ja, jb, jc, jd, jw, k, lj, mj2;
    double wjw[2];

    mj2 = 2 * mj;
    lj = n / mj2;
    int block = lj / process_num;   // 分块大小

    if( block == 0 ){   // lj < process_num，此时问题规模小于进程数，因为block是整型变量，因此截掉小数点后的部分后值为0
        // 此时按照串行的蝶形运算处理
        for (j = 0; j < lj; j++){
            jw = j * mj;
            ja = jw;
            jb = ja;
            jc = j * mj2;
            jd = jc;

            wjw[0] = w[jw * 2 + 0];
            wjw[1] = w[jw * 2 + 1];

            if (sgn < 0.0)
            {
                wjw[1] = -wjw[1];
            }

            for (k = 0; k < mj; k++)
            {
                c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0]; // a和b的实部和
                c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1]; // a和b的虚部和

                ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0]; // a和b的实部差
                ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1]; // a和b的虚部差

                d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu; // 求结果的实部
                d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu; // 求结果的虚部
            }
        }
    }
    else{
        // 分块进行蝶形运算
        for(j = myrank * block; j < ( myrank + 1 ) * block; ++j){
            jw = j * mj;
            ja = jw;
            jb = ja;
            jc = j * mj2;
            jd = jc;

            wjw[0] = w[jw * 2 + 0];
            wjw[1] = w[jw * 2 + 1];

            if (sgn < 0.0)
            {
                wjw[1] = -wjw[1];
            }

            for (k = 0; k < mj; k++)
            {
                c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0]; // a和b的实部和
                c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1]; // a和b的虚部和

                ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0]; // a和b的实部差
                ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1]; // a和b的虚部差

                d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu; // 求结果的实部
                d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu; // 求结果的虚部
            }        

            // 接下来进行进程间的通信    
            if( myrank ){   // 非0进程将运算结果发送给0号进程
                MPI_Send(&c[jc * 2], 2 * mj, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);     // 发送数组c
                MPI_Send(&d[jd * 2], 2 * mj, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);     // 发送数组d
            }
            else{   // 0号进程接收来自其他进程的运算结果
                for(int i = 1; i < process_num; ++i){
                    MPI_Recv(&c[( (i * block) + j ) * mj2 * 2], 2 * mj, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&d[( (i * block) + j ) * mj2 * 2], 2 * mj, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);    // 阻塞，保证其他进程都把运算结果都发送到0号进程
        }
    }
    return;
}

void timestamp() // 记录时间戳
{
#define TIME_SIZE 40

    static char time_buffer[TIME_SIZE];
    const struct tm *tm;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

    cout << time_buffer << "\n";

    return;
#undef TIME_SIZE
}
