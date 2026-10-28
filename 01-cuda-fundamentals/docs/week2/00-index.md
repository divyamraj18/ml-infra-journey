# Week 2

Read in this order. Each file is one day. Assignment 1.2 CUDA code is not written yet.

| File | Question it answers |
|---|---|
| [day8.md](day8.md) | How do 8 GPUs inside one box talk, and why does tiled GEMM load data into SRAM? |
| [day9.md](day9.md) | What are InfiniBand and RoCE, and why can't TCP carry gradients? |
| [day10.md](day10.md) | Where does the CPU still touch the payload on an RDMA / GPUDirect path? |
| [day11.md](day11.md) | Which parallelism uses AllReduce, AllGather, and ReduceScatter, and how many bytes does a ring move? |
| [day12.md](day12.md) | Which `NCCL_DEBUG=INFO` line means you fell back to TCP sockets? |
| [day13-14.md](day13-14.md) | On 4+4 GPUs, when does an AllReduce chunk stay on NVLink, and when does it cross a rail? |

Week 1 coalescing is in [day4](../week1/day4.md). Day 8 extends that 128-byte sector with alignment, stride, and bank conflicts.
