# Bitonic Sort on CUDA

The codes `kernel.cu` aim to implement the bitonic sort algorithm on a GPU to achieve parallel computing. 

## Implementation
1 Calculate the size of array in GPU. Round the size of input array to the nearest power of 2. 

2 Copy the original data to GPU’s global memory. Initialize extra elements with large number 1001.

3 Calculate the proper block size **B**, gride size **G** and number of elements each thread needs to handle **T**.  

4 Kernel function `BlockBitonicSortCuda()`: Using shared memory in each block to sort. Split the whole array into **G** blocks, each block sorts its portion of the array using shared memory. Blocks with even indices sort their elements in non-decreasing order. Blocks with odd indices sort their elements in non-increasing order.

5 Kernel function `BitoincSortCuda()`: Merge the sorted subarray in step 4 into one. Each thread performs a single comparison of two element in the GPU’s global memory. The outer loop and inner loop of the sort all run in CPU. For each iteration of the inner loop, the kernel function is called to perform comparisons and swaps. 

6 Copy the sorted array from the GPU's global memory back to the host memory. 

## Optimization Strategies
The total complexity of bitonic sort is O(n \log^2 n) for serialized computing. 

Several techniques are performed to improve its performance:

1) The bitonic sort is easy to be parallelled. Leveraging GPU’s parallel computing, we can assign each thread to do the comparison concurrently. This approach reduces the overall processing time by maximizing computational throughput. 

2) Decrease the I/O memory accesses. As the I/O operation is expensive, shared memory which is in chip is utilized. The first kernel function utilizes shared memory to sort subarrays within a block. Each thread then manages its own shared memory for reading and writing during sorting. 

    However, there are limitations with shared memory as its maximum size is 48 KB. As the subarray size increases, individual threads perform more serial comparisons and swaps, impacting concurrent processing. Consequently, a secondary function is invoked to merge all sorted subarrays within each block. In function2, each thread performs one time comparison and swap which achieves high parallelism. 

3) Since invoking kernel functions incurs overhead, it's essential to minimize their frequency. This optimization strategy aims to reduce the time spent on GPU hardware allocation.

4) Employing bitwise operations instead of costly divide and modulo operations enhances computational efficiency.

 By implementing these optimization techniques, the Bitonic Sort algorithm can achieve significantly improved efficiency. The primary bottleneck of current implementation lies in merging sorted subarrays as function `BitoincSortCuda()` is called for every inner iteration loop.  Implementing alternative algorithms or optimizations specifically tailored for this merging process may further enhance overall performance.

