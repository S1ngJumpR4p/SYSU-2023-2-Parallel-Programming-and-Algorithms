# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <omp.h>
# include <mpi.h>
# include <vector>
# include <numeric>
using namespace std;

int main ( );
void ccopy ( int n, double x[], double y[] );
void cfft2 ( int n, double x[], double y[], double w[], double sgn );
void cffti ( int n, double w[] );
double cpu_time ( void );
double ggl ( double *ds );
void step ( int n, int mj, double a[], double b[], double c[], double d[], 
  double w[], double sgn);
void timestamp ( );

int myrank, process_num;  

int main ( ){
  double ctime;
  double ctime1;
  double ctime2;
  double error;
  int first;
  double flops;
  double fnm1;
  int i;
  int icase;
  int it;
  int ln2;
  double mflops;
  int n;
  int nits = 10000;
  static double seed;
  double sgn;
  double *w;
  double *x;
  double *y;
  double *z;
  double z0;
  double z1;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_num);
  if(myrank == 0){
    timestamp ( );
    cout << "\n";
    cout << "FFT_SERIAL\n";
    cout << "  C++ version\n";
    cout << "\n";
    cout << "  Demonstrate an implementation of the Fast Fourier Transform\n";
    cout << "  of a complex data vector.\n";
    cout << "\n";
    cout << "  Accuracy check:\n";
    cout << "\n";
    cout << "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n";
    cout << "\n";
    cout << "             N      NITS    Error         Time          Time/Call     MFLOPS\n";
    cout << "\n";
  }
  seed  = 331.0;
  n = 1;
  for ( ln2 = 1; ln2 <= 20; ln2++ )
  {
    n = 2 * n;
    w = new double[  n];
    x = new double[2*n];
    y = new double[2*n];
    z = new double[2*n];

    first = 1;

    for ( icase = 0; icase < 2; icase++ ){
      if ( first ){
        for ( i = 0; i < 2 * n; i = i + 2 ){
          z0 = ggl ( &seed );
          z1 = ggl ( &seed );
          x[i] = z0;
          z[i] = z0;
          x[i+1] = z1;
          z[i+1] = z1;
        }
      } 
      else{
        for ( i = 0; i < 2 * n; i = i + 2 ){
          z0 = 0.0;
          z1 = 0.0;
          x[i] = z0;
          z[i] = z0;
          x[i+1] = z1;
          z[i+1] = z1;
        }
      }
      cffti ( n, w );
      if ( first ){
        sgn = + 1.0;
        cfft2 ( n, x, y, w, sgn );  // 先进行DFT
        sgn = - 1.0;
        cfft2 ( n, y, x, w, sgn );  // 再进行IDFT
        fnm1 = 1.0 / ( double ) n;
        error = 0.0;
        for ( i = 0; i < 2 * n; i = i + 2 ){
          error = error + pow ( z[i]   - fnm1 * x[i], 2 ) + pow ( z[i+1] - fnm1 * x[i+1], 2 );
        }
        error = sqrt ( fnm1 * error );
        if(myrank == 0)
          cout << "  " << setw(12) << n << "  " << setw(8) << nits << "  " << setw(12) << error;
        first = 0;
      }
      else{
        ctime1 = cpu_time ( );
        for ( it = 0; it < nits; it++ ){
          sgn = + 1.0;
          cfft2 ( n, x, y, w, sgn );
          sgn = - 1.0;
          cfft2 ( n, y, x, w, sgn );
        }
        ctime2 = cpu_time ( );
        ctime = ctime2 - ctime1;
        flops = 2.0 * ( double ) nits * ( 5.0 * ( double ) n * ( double ) ln2 );
        mflops = flops / 1.0E+06 / ctime;
        if(myrank == 0)
          cout << "  " << setw(12) << ctime << "  " << setw(12) << ctime / ( double ) ( 2 * nits ) << "  " << setw(12) << mflops << "\n";
      }
    }
    if ( ( ln2 % 4 ) == 0 ) {
      nits = nits / 10;
    }
    if ( nits < 1 ) {
      nits = 1;
    }
    delete [] w;
    delete [] x;
    delete [] y;
    delete [] z;
  }
  if(myrank == 0){
    cout << "\n" << "FFT_SERIAL:\n" << "  Normal end of execution.\n" << "\n";
    timestamp ( );
  }
  MPI_Finalize();
  return 0;
}

void ccopy ( int n, double x[], double y[] ) {
  int i;
  for ( i = 0; i < n; i++ ){
    y[i*2+0] = x[i*2+0];
    y[i*2+1] = x[i*2+1];
   }
  return;
}

void cfft2 ( int n, double x[], double y[], double w[], double sgn ){
  int j;
  int m;
  int mj;
  int tgle;
  m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) ); // 分解深度，蝶形算法的复杂度为O(NlogN)
  mj   = 1;
  tgle = 1;
  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );

  if ( n == 2 ){
    return;
  }
  for ( j = 0; j < m - 2; j++ ){
    mj = mj * 2;
    if ( tgle ){
      step ( n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w, sgn );
      tgle = 0;
    }
    else{
      step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
      tgle = 1;
    }
  }
  if ( tgle ) {
    ccopy ( n, y, x );
  }
  mj = n / 2;
  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn);
  return;
}

void cffti ( int n, double w[] )  {
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;
  n2 = n / 2;
  aw = 2.0 * pi / ( ( double ) n );
  for ( i = 0; i < n2; i++ ){
    arg = aw * ( ( double ) i );
    w[i*2+0] = cos ( arg );
    w[i*2+1] = sin ( arg );
  }
  return;
}

double cpu_time ( void ){
  double value;
  value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;
  return value;
}

double ggl ( double *seed ){
  double d2 = 0.2147483647e10;
  double t;
  double value;
  t = *seed;
  t = fmod ( 16807.0 * t, d2 );
  *seed = t;
  value = ( t - 1.0 ) / ( d2 - 1.0 );
  return value;
}

void step ( int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn){
  double ambr, ambu;
  int j, ja, jb, jc, jd, jw, k, lj, mj2; 
  double wjw[2]; 
  mj2 = 2 * mj;
  lj = n / mj2;
  if(n < process_num){    // 问题规模太小，串行处理即可
    for ( j = 0; j < lj; j++ ){
      jw = j * mj;
      ja  = jw;
      jb  = ja;
      jc  = j * mj2;
      jd  = jc;
      wjw[0] = w[jw*2+0]; 
      wjw[1] = w[jw*2+1];
      if ( sgn < 0.0 ) {
        wjw[1] = - wjw[1];
      }
      for ( k = 0; k < mj; k++ ){
        c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];  // a和b的实部和
        c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];  // a和b的虚部和
        ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0]; // a和b的实部差
        ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1]; // a和b的虚部差
        d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;  // 求结果的实部
        d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;  // 求结果的虚部
      }
    }
  }
  else{   // 问题规模较大，可能可以进行并行化处理
    if(lj >= mj2){
      if(lj/process_num < 1){   // 分的块太小，串行处理
        for ( j = 0; j < lj; j++ ){
          jw = j * mj;
          ja  = jw;
          jb  = ja;
          jc  = j * mj2;
          jd  = jc;
          wjw[0] = w[jw*2+0]; 
          wjw[1] = w[jw*2+1];
          if ( sgn < 0.0 ) {
            wjw[1] = - wjw[1];
          }
          for ( k = 0; k < mj; k++ ){
            c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];  // a和b的实部和
            c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];  // a和b的虚部和
            ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0]; // a和b的实部差
            ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1]; // a和b的虚部差
            d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;  // 求结果的实部
            d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;  // 求结果的虚部
          }
        }        
      }
      else{
        int chunkSize = (int)lj / process_num;
        vector<int> lj_indices(lj);
        vector<int> recv_indices(chunkSize);
        iota(lj_indices.begin(), lj_indices.end(), 0);
        int c_length = lj * mj * 2;
        double* _c  = new double[c_length];
        int c_chunkSize = c_length /  process_num;
        int d_length = lj * mj * 2;
        double* _d = new double[d_length];
        int d_chunkSize = d_length /  process_num;
        if(myrank==0){
          MPI_Scatter(lj_indices.data(), chunkSize, MPI_INT, recv_indices.data(), chunkSize, MPI_INT, 0, MPI_COMM_WORLD);
          MPI_Gather(_c, c_chunkSize, MPI_DOUBLE, c, c_chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          MPI_Gather(_d, d_chunkSize, MPI_DOUBLE, d, d_chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);     
        }
        else{
          MPI_Scatter(lj_indices.data(), chunkSize, MPI_INT, recv_indices.data(), chunkSize, MPI_INT, 0, MPI_COMM_WORLD);
          int _jw, _ja, _jb, _jc, _jd;
          double w0, w1, _ambr, _ambu;
          for(auto iter = recv_indices.begin(); iter != recv_indices.end(); ++iter){
            int p = *iter;
            _jw = p * mj;
             _ja = _jw;
              _jb = _jw;
            _jc = p * mj2;
             _jd = p * mj2;
            w0 = w[_jw*2+0];
            w1 = w[_jw*2+1];
            if(sgn < 0){
              w1 = -w1;
            }
            for (int q = 0; q < mj; q++ ){
              _c[(_jc+q)*2+0] = a[(_ja+q)*2+0] + b[(_jb+q)*2+0];  // a和b的实部和
              _c[(_jc+q)*2+1] = a[(_ja+q)*2+1] + b[(_jb+q)*2+1];  // a和b的虚部和
              _ambr = a[(_ja+q)*2+0] - b[(_jb+q)*2+0]; // a和b的实部差
              _ambu = a[(_ja+q)*2+1] - b[(_jb+q)*2+1]; // a和b的虚部差
              _d[(_jd+q)*2+0] = w0 * _ambr - w1 * _ambu;  // 求结果的实部
              _d[(_jd+q)*2+1] = w1 * _ambr + w0 * _ambu;  // 求结果的虚部
            }
          }
          MPI_Gather(_c, c_chunkSize, MPI_DOUBLE, c, c_chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          MPI_Gather(_d, d_chunkSize, MPI_DOUBLE, d, d_chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        delete []_c;
        delete []_d;     
      }
    }
    else{
      if(mj / process_num < 1){
        for ( j = 0; j < lj; j++ ){
          jw = j * mj;
          ja  = jw;
          jb  = ja;
          jc  = j * mj2;
          jd  = jc;
          wjw[0] = w[jw*2+0]; 
          wjw[1] = w[jw*2+1];
          if ( sgn < 0.0 ) {
            wjw[1] = - wjw[1];
          }
          for ( k = 0; k < mj; k++ ){
            c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];  // a和b的实部和
            c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];  // a和b的虚部和
            ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0]; // a和b的实部差
            ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1]; // a和b的虚部差
            d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;  // 求结果的实部
            d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;  // 求结果的虚部
          }
        }        
      }
      else{
        int chunkSize = (int)mj / process_num;
        vector<int> mj_indices(mj);
        vector<int> recv_indices(chunkSize);
        iota(mj_indices.begin(), mj_indices.end(), 0);
        int c_length = lj * mj * 2;
        double* _c  = new double[c_length];
        int c_chunkSize = c_length /  process_num;
        int d_length = lj * mj * 2;
        double* _d = new double[d_length];
        int d_chunkSize = d_length /  process_num;
        if(myrank==0){
          MPI_Scatter(mj_indices.data(), chunkSize, MPI_INT, recv_indices.data(), chunkSize, MPI_INT, 0, MPI_COMM_WORLD);
          MPI_Gather(_c, c_chunkSize, MPI_DOUBLE, c, c_chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          MPI_Gather(_d, d_chunkSize, MPI_DOUBLE, d, d_chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        else{
          MPI_Scatter(mj_indices.data(), chunkSize, MPI_INT, recv_indices.data(), chunkSize, MPI_INT, 0, MPI_COMM_WORLD);
          int _jw, _ja, _jb, _jc, _jd;
          double w0, w1, _ambr, _ambu;
          for(int p = 0; p < lj; ++p){
            _jw = p * mj;
            _ja = _jw;
             _jb = _jw;
            _jc = p * mj2;
            _jd = p * mj2;
            w0 = w[_jw*2+0];
            w1 = w[_jw*2+1];
            if(sgn < 0){
              w1 = -w1;
            }
            for(auto iter = recv_indices.begin(); iter != recv_indices.end(); ++iter){
              int q = *iter;
              _c[(_jc+q)*2+0] = a[(_ja+q)*2+0] + b[(_jb+q)*2+0];  // a和b的实部和
              _c[(_jc+q)*2+1] = a[(_ja+q)*2+1] + b[(_jb+q)*2+1];  // a和b的虚部和
              _ambr = a[(_ja+q)*2+0] - b[(_jb+q)*2+0]; // a和b的实部差
              _ambu = a[(_ja+q)*2+1] - b[(_jb+q)*2+1]; // a和b的虚部差
              _d[(_jd+q)*2+0] = w0 * _ambr - w1 * _ambu;  // 求结果的实部
              _d[(_jd+q)*2+1] = w1 * _ambr + w0 * _ambu;  // 求结果的虚部              
            }            
          }          
          MPI_Gather(_c, c_chunkSize, MPI_DOUBLE, c, c_chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          MPI_Gather(_d, d_chunkSize, MPI_DOUBLE, d, d_chunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);          
        }
        delete []_c;
        delete []_d;   
      }
    }
  }
  return;
}

void timestamp ( ){
# define TIME_SIZE 40
  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;
  now = time ( NULL );
  tm = localtime ( &now );
  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );
  cout << time_buffer << "\n";
  return;
# undef TIME_SIZE
}

