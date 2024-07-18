#include<cudnn.h>
#include<random>
#include<cstdio>
#include<iostream>
#include<fstream>
using namespace std;
#define Check(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << endl; \
      exit(EXIT_FAILURE);                               \
    }                                                        \
  }


__host__ void init(float* mat, int row, int col) {
    // 初始化随机数生成器
    random_device rd;
    default_random_engine eng(rd());
    uniform_int_distribution<int> distr(0, 255);

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            mat[i * col + j] = distr(eng);
        }
    }
}



int main(){
    
    int w, h, w_out, h_out, f_w = 3, f_h = 3, stride;
    printf("请输入图像大小：\n");
    scanf("%d %d", &h, &w);
    
    printf("请输入步长：\n");
    scanf("%d", &stride);
    
    printf("请输入输出图像大小：\n");
    scanf("%d %d", &h_out, &w_out);
    
    cudnnHandle_t handle;
    Check(cudnnCreate(&handle));
    float* filter;
    cudaMallocHost(&filter, f_w * f_h * 3 * 3 * sizeof(float));
    for(int i = 0; i < f_w*f_h*3*3; ++i){
        filter[i] = 0.01;
    }
    int imageBytes = w * h * 3 * sizeof(float);
    float *image;
    cudaMallocHost(&image, imageBytes);
    init(image, h * 3, w);
    
    int delta = (w_out - 1) * stride - w + f_w;
    int pad = (delta + 1) / 2;
    float* res;
    int outBytes = w_out * h_out * 3 * sizeof(float);
    cudaMallocHost(&res, outBytes);
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    Check(cudnnCreateTensorDescriptor(&input_desc));
    Check(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 3, h, w));
    
    Check(cudnnCreateFilterDescriptor(&filter_desc));
    Check(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, 3, 3, 3));
    
    Check(cudnnCreateConvolutionDescriptor(&conv_desc));
    Check(cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    
    Check(cudnnCreateTensorDescriptor(&output_desc));
    Check(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 3, h_out, w_out));
    
    cudnnConvolutionFwdAlgo_t conv_algorithm;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t conv_algorithm_perf;
    Check(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc, filter_desc, conv_desc, output_desc, 1, &returnedAlgoCount, &conv_algorithm_perf));
    conv_algorithm = conv_algorithm_perf.algo;
    
    size_t workspaceBytes = 0;
    Check(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, conv_algorithm, &workspaceBytes));
    
    void* workspace;
    cudaMallocHost(&workspace, workspaceBytes);
    
    float alpha = 1, beta = 0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    
    Check(cudnnConvolutionForward(handle, &alpha, input_desc, image, filter_desc, filter, conv_desc, conv_algorithm, workspace, workspaceBytes, &beta, output_desc, res));
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);
    printf("图像大小:%d*%d;所用时间:%f ms\n",h,w,ms);    
    string output_name = to_string(h)+"*"+to_string(w)+"_cuDNN.txt";
    ofstream f(output_name);
    if(f.is_open()){
        for(int i = 0 ; i < 3; ++i){
            for(int j = 0; j < h_out; ++j){
                for(int k = 0; k < w_out; ++k){
                    if(res[j * w_out + k] > 255){
                        res[j * w_out + k] = 255;
                    }
                    if(res[j * w_out + k] < 0){
                        res[j * w_out + k] = 0;
                    }
                    f << res[j * w_out + k]<<" ";
                }
                f << "\n";
            }
            f << "\n";
        }
    }
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudaFreeHost(workspace);
    cudaFreeHost(res);
    cudaFreeHost(image);
    cudaFreeHost(filter);
    cudnnDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return 0;
}
