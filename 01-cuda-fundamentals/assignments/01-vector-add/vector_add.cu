// Assignment 1.1 — vector add
//
// Colab T4 (compute capability 7.5):
//   nvcc -arch=sm_75 vector_add.cu -o vector_add && ./vector_add
//
// Host = CPU + DRAM. Device = GPU + HBM.
// Kernel = function that the GPU runs as a grid of threads.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

// cudaMalloc / cudaMemcpy return cudaError_t. cudaSuccess ke alawa
// koi bhi code yahin ruk jaata hai, warna baad ka segfault padhta hai
// aur asli error (OOM, bad copy) chhup jaata hai.
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Har thread ek index i pe c[i] = a[i] + b[i] karta hai.
// i = blockIdx.x * blockDim.x + threadIdx.x
// Thread i element i padhta hai, isliye warp ka access coalesced hai:
// 32 threads, 32 consecutive floats, ek 128-byte HBM transaction.
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Grid last block mein extra threads launch karti hai jab n block size
  // ka multiple na ho. Un threads ko HBM se bahar nahi padhne dete.
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int n = 1 << 20;  // 1,048,576 elements
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  // Host buffers. std::vector DRAM mein hai.
  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_c(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  // Device pointers HBM ko point karte hain. cudaMalloc CPU pe pointer
  // return karta hai, memory GPU pe allocate hoti hai.
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  // cudaMemcpyHostToDevice = DRAM -> PCIe -> HBM. Kernel is copy ka
  // intezaar karta hai kyunki default stream synchronous hai.
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  // 256 = 8 warps. 32 ka multiple, isliye last warp mein seats khali nahi
  // jaati. Grid utne blocks ki hai ki n threads cover ho jayein.
  const int threads_per_block = 256;
  const int blocks =
      (n + threads_per_block - 1) / threads_per_block;

  // cudaEvent GPU timeline pe timestamp hai. CPU clock PCIe aur launch
  // overhead mix kar deta hai. Events sirf kernel ko naapne ke liye hain.
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  vectorAdd<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
  // <<<>>> khud cudaError_t return nahi karta. Launch error alag se lo.
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  // Record async hai. Elapsed time se pehle stop event complete hona chahiye.
  CUDA_CHECK(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    const float expected = h_a[i] + h_b[i];
    if (h_c[i] != expected) {
      ++mismatches;
      if (mismatches < 5) {
        std::printf("mismatch i=%d got %f expected %f\n", i, h_c[i],
                    expected);
      }
    }
  }

  // Kernel 2 arrays padhta hai aur 1 likhta hai. Teen HBM trips.
  // GB/s = bytes / seconds. milliseconds ko seconds mein: / 1000.
  const double moved_bytes = 3.0 * static_cast<double>(bytes);
  const double seconds = static_cast<double>(milliseconds) / 1000.0;
  const double gib_per_s =
      seconds > 0.0 ? (moved_bytes / seconds) / (1024.0 * 1024.0 * 1024.0)
                    : 0.0;

  std::printf("n=%d blocks=%d threads_per_block=%d\n", n, blocks,
              threads_per_block);
  std::printf("kernel %.3f ms, effective %.2f GiB/s, mismatches %d\n",
              milliseconds, gib_per_s, mismatches);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
