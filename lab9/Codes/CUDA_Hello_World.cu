# include <stdio.h>

__global__ void PrintHelloWorld(void){
    int blockID = blockIdx.x;
    int x = threadIdx.x;
    int y = threadIdx.y;
    printf("Hello World from Thread (%d, %d) in Block %d!\n",x,y,blockID);
}

int main(){
    int n, m, k;
    printf("请输入n、m、k:\n");
    scanf("%d %d %d",&n,&m,&k);
    printf("Hello World from the host!\n");
    PrintHelloWorld<<<n,dim3(m,k)>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}