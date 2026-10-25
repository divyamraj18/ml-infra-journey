# Week 2

Padhne ka order. Har file ek din hai. Assignment 1.2 ka CUDA code abhi nahi hai.

| File | Sawaal |
|---|---|
| [day8.md](day8.md) | Ek box ke andar 8 GPUs kaise jude hain, aur tiled GEMM SRAM mein data kyun lata hai? |
| [day9.md](day9.md) | Box ke bahar InfiniBand aur RoCE kya hain, aur TCP gradient kyun nahi le ja sakta? |
| [day10.md](day10.md) | RDMA aur GPUDirect pe CPU payload ko touch kahaan karta hai? |
| [day11.md](day11.md) | AllReduce, AllGather, ReduceScatter kis parallelism ka hai, aur ring pe kitne bytes jaate hain? |
| [day12.md](day12.md) | `NCCL_DEBUG=INFO` mein kaun si line batati hai ki tum Socket pe gir gaye? |
| [day13-14.md](day13-14.md) | 4+4 GPUs pe AllReduce ka chunk NVLink pe rehta hai aur rail pe kab chadhta hai? |

Week 1 ka coalescing [day4](../week1/day4.md) hai. Day 8 usi 128-byte sector ko alignment, stride, aur bank conflicts tak le jaata hai.
