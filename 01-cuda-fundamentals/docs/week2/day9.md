# Day 9: InfiniBand, RoCE, and why TCP is too slow

Day 8 ends at the box. The last NVSwitch port stops at the chassis. Traffic between two DGX machines is **inter-node**. That needs a NIC and a separate fabric.

## The TCP/IP path

The gradient on the GPU lives in HBM. TCP sends through a socket, and the socket lives in CPU DRAM. Without RDMA the path is:

```text
GPU0 HBM
  -> cudaMemcpy over PCIe
CPU0 DRAM
  -> kernel TCP stack: copy, checksum, context switch
NIC0
  -> wire
NIC1
  -> kernel TCP
CPU1 DRAM
  -> cudaMemcpy over PCIe
GPU1 HBM
```

The payload crosses PCIe twice, and the CPU kernel sits in the middle. Every small message is a syscall. Latency is tens to hundreds of microseconds. Bandwidth stalls on the CPU and on PCIe, not on the NIC's wire rate.

LLM training does this every step. Data parallel runs an **AllReduce** every step: sum every GPU's gradients, then give that same sum back to every GPU. A 7B model in BF16 has `7e9 × 2 bytes = 14 GB` of gradients. Those 14 GB move across the fabric every step.

Ring AllReduce, covered properly on Day 11, sends about `2 × S` bytes when there are many GPUs. One NIC at 400 Gb/s:

```text
S = 14 GB = 112 Gb
2S / 400 Gb/s ≈ 0.56 s
```

This is a single-rail sketch with no overlap. It is an upper bound, not a measured step time. Real NCCL does better because a node has 8 NICs and the collective overlaps with compute. The point is that TCP makes this number worse, because TCP never reaches 400 Gb/s. If the compute step is 100 ms and the network takes 500 ms, the GPU sits idle for those 500 ms. Same stall as Day 2, except the wait is the network instead of HBM.

## InfiniBand

**InfiniBand (IB)** is an HPC fabric. It is not Ethernet. The link layer is **credit-based**: the sender puts a packet on the wire only after the receiver has granted credits. Traffic stops before a switch buffer overflows. Dropping packets is not part of the design. There is no TCP-style drop and retransmit.

RDMA is native on InfiniBand. The NIC writes straight into registered memory on the remote machine. The CPU does not touch that payload.

Generations you will see in clusters:

| Generation | Per-port rate | NIC that usually comes with it |
|---|---|---|
| HDR | 200 Gb/s | ConnectX-6 |
| NDR | 400 Gb/s | ConnectX-7 |
| XDR | 800 Gb/s | next generation |

A DGX H100 typically has **8 ConnectX-7** NICs, each 400 Gb/s NDR, and **Quantum-2** InfiniBand switches. There are 8 NICs because there are 8 GPUs: one NIC per GPU, one **rail**. A rail becomes a topology on Day 12.

Small messages land in the ~1 microsecond class, against tens of microseconds for TCP. Large gradients are limited by the port bandwidth, not by a syscall.

## RoCE

**RoCE** means RDMA over Converged Ethernet. The RDMA verbs (queue pair, write, read) are the same. The wire is Ethernet.

**RoCE v2** is wrapped in UDP/IP, so it can cross routers. The UDP port is **4791**. RoCE v1 stayed inside one L2 subnet. Clusters use v2.

Ethernet can drop packets by default. RDMA does not want drops, because it does not have TCP's reliable byte stream. RoCE therefore needs a managed fabric:

- **PFC** (Priority Flow Control) pauses one priority with a pause frame, not the whole link. That reduces loss. A bad config causes a **PFC storm** and head-of-line blocking. The whole cluster can stall on pause frames.
- **ECN** lets the switch mark a packet before it drops it.
- **DCQCN** is the sender's reaction to ECN: it cuts its rate. That is RoCE congestion control.

InfiniBand gets the same property from credits. RoCE gets it from switch config. With a correct config, bandwidth is close to InfiniBand. With a bad config, training still produces the right loss and the step time doubles. Debugging that is Day 12.

## Which fabric

| | InfiniBand | RoCE v2 |
|---|---|---|
| Wire | Its own IB fabric | Ethernet |
| Loss | Credits, no drops | Avoid drops with PFC + ECN + DCQCN |
| RDMA | Native | Same verbs, on UDP |
| Ops cost | Separate switches, separate tooling | Ethernet you may already run, but PFC is fragile |
| Where you see it | DGX SuperPOD, NVIDIA reference designs | Many cloud GPU clusters |

Both data paths are RDMA. Neither is TCP. If an NCCL log says `Socket`, you are on neither of them. That is the first check on Day 12.

## Rule for today

NVSwitch stays inside the box. Outside the box the fabric is **InfiniBand or RoCE**, and both are RDMA. TCP drags a GPU gradient through CPU DRAM and the kernel, so the GPU waits on the network during the step.
