# How Software Maps Directly onto These Cores

## Mapping Threads to Data: The Indexing Formula

When you launch a GPU kernel, you define a grid of thread blocks. Every thread receives unique built-in coordinate variables from the hardware:

1. `blockIdx.x` — Tells the thread which block it belongs to within the grid.
2. `blockDim.x` — Tells you how many threads are packed inside each block.
3. `threadIdx.x` — Tells the thread its exact local position within its block.

To find the unique global data index for any thread across the entire array, we combine these variables into a 1D indexing formula:

```cpp
idx = blockIdx.x * blockDim.x + threadIdx.x
```

**Example:** If each block contains 256 threads, and you are looking at thread index 5 inside block 1, the global data index is calculated as:

```text
1 * 256 + 5 = 261
```

This thread will load, compute, and store data exclusively at position 261 in global memory.

## Array Bounds and Grid Sizing

In most real-world scenarios, the total data size (`N`) will not be a clean, perfect multiple of our block size (e.g. 256 threads). If we launch just enough blocks to cover `N`, our last block might have extra active threads that will exceed our array size.

If those extra threads try to read or write memory outside our array bounds, our GPU application will crash or cause illegal memory access errors.

To prevent this, every kernel needs a safety check before doing any work:

```cpp
if (idx < N) {
    // Perform computation
}
```

## Host–Device Memory Management

1. **CPU ki memory se GPU ki memory allocate karna (`cudaMalloc`)**

   Jaise hum C++ mein `malloc` ya `new` keyword use karke RAM mein memory allocate karte hain, similarly GPU ke VRAM mein memory allocate karne ke liye `cudaMalloc` use karte hain.

   **How it works?**  
   Isme hum GPU pointer ka address aur jitna size chahiye wo pass karte hain. Aur jaise hi ye run hota hai, GPU apne memory mein utni jagah reserve kar leta hai.

2. **Data ko CPU se GPU mein bhejna (`cudaMemcpyHostToDevice`)**

   CPU ki RAM mein jo data rakha hai (e.g. array, matrix), usko GPU ke VRAM mein copy karna padta hai taaki CUDA cores us par parallel calculation kar sakein.

3. **GPU par kernel run karna (computation)**

   Ek baar data GPU memory mein pahunch gaya, tab hum apna CUDA kernel launch karte hain. Saare threads apne-apne index par parallel calculations perform karte hain aur output ko GPU ki memory mein hi store karte hain.

4. **Output ko wapas GPU se CPU mein lana (`cudaMemcpyDeviceToHost`)**

   Calculation khatam hone ke baad final result GPU ki VRAM mein hota hai. Agar hume usko print karna hai, toh wapas copy karna padta hai.

5. **Memory free karna taaki leak na ho (`cudaFree`)**

   Jaise C++ mein `free` ya `delete` use karke memory deallocate karte hain, usi tarah se `cudaFree` use karke GPU memory allocation ko release kar dete hain.

## CUDA Events and Execution Timing

So when we write code in C++ we use the standard timer library (`std::chrono`) to measure how long a function takes. But if we try that on the GPU, it will give us completely wrong numbers.

### Why? The asynchronous problem

- When the CPU tells the GPU to run a kernel, the CPU doesn't wait around for the GPU to finish the task. It immediately moves to the next line of code.
- If we use a standard timer right around the kernel launch, the timer will stop immediately, kyuki CPU ne kernel launch ka kaam khatam kar liya, although ki kaam actual mein ho raha hai backend mein GPU mein.

### How to solve? CUDA Events (`cudaEvent_t`)

- CUDA provides special hardware-level timers called events that live directly inside the GPU execution stream.
- They record an exact timestamp on the GPU silicon itself, bypassing the CPU async delay.

### How CUDA events work in practice

To time a kernel properly, we follow a 4-step sequence:

1. **Create the events:**

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
```

2. **Record the start time:**

```cpp
cudaEventRecord(start, 0); // Drops a timestamp right before the kernel
```

3. **Launch the kernel:**

```cpp
my_kernel<<<grid, block>>>(...);
```

4. **Record the stop time and calculate elapsed time:**

```cpp
cudaEventRecord(stop, 0);       // Drops a timestamp right after the kernel finishes
cudaEventSynchronize(stop);     // Forces CPU to wait until the stop event is reached

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```





### Robust Error Handling

In CUDA development, bugs can hide easily because many operations run asynchronously. If a memory allocation fails or a kernel launch configuration is invalid, our program might not error out right away. Rather, it might fail later with a cryptic segmentation fault, making it hard to debug.

**How to solve?**

1. **API error codes** — Standard runtime functions like `cudaMalloc` or `cudaMemcpy` return a status code of type `cudaError_t`. We can check if the code equals `cudaSuccess`.

2. **Async kernel errors** — Because kernels run asynchronously and don't return a status code directly, we must call `cudaGetLastError()` right after a kernel launch to catch immediate configuration mistakes (such as passing an invalid block size).

To make this clean, developers use a wrapper macro like this:

```cpp
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)
```