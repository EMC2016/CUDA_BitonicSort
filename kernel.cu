#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

//**OPT5: Bit Operations replace modulo and division operations.
__global__ void BlockBitonicSortCuda(int* A,int size, int threadSize){
    extern __shared__ int shareA[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i< threadSize; i++){
        if ((idx*threadSize+i) >= size){
            A[idx*threadSize+i] = 1001;
        }
        shareA[threadIdx.x*threadSize+i] = A[idx*threadSize+i];
    }
    
    __syncthreads();

    int k;
    int n = blockDim.x * threadSize;
    int index;
    int temp;
    for (int i = 2;i<=n;i *=2){
        for (int j = i>>1; j>0; j >>=1){
            for (int c = 0; c < threadSize; c++){
                k = threadIdx.x * threadSize +c;
                index= k^j;
                if(index>k) {
                    if (((i&k)==0 && (blockIdx.x&1)==0)||(((i&k)!=0 && (blockIdx.x&1)!=0))){
                        if(shareA[k]>shareA[index]){
                            
                            temp = shareA[k];
                            shareA[k] = shareA[index];
                            shareA[index] = temp;
                        }
                    }else{
                        if(shareA[k]<shareA[index]){
                            
                            temp = shareA[k];
                            shareA[k] = shareA[index];
                            shareA[index] = temp;
                        }
                    }
                }
            }   
            __syncthreads();
        }
        
    }
    __syncthreads();

    
    for (int i = 0; i< threadSize; i++){
        A[idx*threadSize+i] = shareA[threadIdx.x*threadSize+i];
    }
    
}
    

__global__ void BitoincSortCuda(int* A,int i,int j) {   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int k,index,temp;
   
        k = idx;
        index= k^j;
        if(index > k){    
            if ((i&k)==0){
                if(A[k]>A[index]){          
                    temp = A[k];
                    A[k] = A[index];
                    A[index] = temp;
                }
            }else{
                if(A[k]<A[index]){     
                    temp = A[k];
                    A[k] = A[index];
                    A[index] = temp;
                }
            }   

        }
     

}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

     // arCpu contains the input random array.
    // arrSortedGpu should contain the sorted array copied from GPU to CPU.
    //***OPT1: Allocates page-locked memory on the host to sppeedup memcpy from CPU to GPU.
    int* arrCpu;
    cudaMallocHost(&arrCpu, size*sizeof(int));
    int* arrSortedGpu;
    cudaMallocHost(&arrSortedGpu,size*sizeof(int));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    //Adjust size of the array arrCpu and copy from CPU to GPU.
    int *A;
    int n = size;
    n--;           
    n |= n >> 1;  
    n |= n >> 2;   
    n |= n >> 4;  
    n |= n >> 8;
    n |= n >> 16; 
    n++;  
    cudaMallocManaged(&A, n * sizeof(int));
    cudaMemcpy(A, arrCpu, size* sizeof(int), cudaMemcpyHostToDevice);

    //Calculate block and grid size, call kernel function to sort data in each block.
    //***OPT2: Employing shared memory of each block.
    //***OPT3: Threads in each block operate in parallel.
    int blockSize;   
    int minGridSize; 
    int gridSize; 
    int threadSize = 4;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, BlockBitonicSortCuda, 0, 0); 
    
    gridSize = (n/threadSize + blockSize - 1) / blockSize;
    
    BlockBitonicSortCuda<<<gridSize, blockSize, threadSize*blockSize*sizeof(int)>>>(A,size,threadSize);
    cudaDeviceSynchronize();
    
    //Merge sorted data in each block.
    //***OPT4: Threads in each block operate in parallel.
    for (int i = blockSize*threadSize*2;i<=n;i *=2){
        for (int j = i>>1; j>0; j >>=1){
            BitoincSortCuda<<<gridSize*threadSize, blockSize>>>(A,i,j);
            cudaDeviceSynchronize();
        }
    }
    
    //Copy sorted data from GPU to CPU.
    cudaMemcpy(arrSortedGpu, A, size* sizeof(int), cudaMemcpyDeviceToHost);    
    cudaFree(A);
    cudaFreeHost(arrCpu); 
    cudaFreeHost(arrSortedGpu);
    

    return 0;
}
