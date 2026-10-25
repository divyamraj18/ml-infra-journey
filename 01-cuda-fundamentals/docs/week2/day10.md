# Day 10: RDMA

Day 9 ne kaha ki gradient CPU ke TCP stack se nahi guzarna chahiye. **RDMA (Remote Direct Memory Access)** woh mechanism hai. NIC remote machine ki memory mein likhta hai. Remote CPU `recv()` nahi chalaata.

## Do machines, beech mein NIC

Har side pe ek **RNIC** hota hai. DGX pe yeh ConnectX hai. RNIC userspace se registered memory ka physical address jaanta hai, isliye kernel se permission leke DMA karta hai.

Teen objects yaad rakhne hain:

| Object | Kya hai |
|---|---|
| **Queue Pair (QP)** | Send Queue + Receive Queue. Ek connection ka endpoint. TCP socket ka RDMA wala cousin, lekin payload kernel se nahi guzarta. |
| **Completion Queue (CQ)** | NIC yahan likhta hai ki kaun sa work request khatam hua. CPU poll karta hai. Interrupt zaroori nahi. |
| **Memory Region (MR)** | Pages jo NIC ke liye pin ho chuke hain. **lkey** local access ke liye, **rkey** remote writer ke liye. Bina rkey ke remote NIC tumhari memory nahi chhoo sakta. |

**Protection Domain** in keys ko ek group mein band karta hai, taaki doosre process ka QP tumhara MR use na kar le.

Setup pe CPU ab bhi kaam karti hai: pages pin karo, QP banao, remote ko apna rkey aur address bhejo. Yeh control messages chhote hote hain. **Bootstrap** aksar TCP se hota hai. Data path uske baad RDMA hai. Bootstrap ko data path mat samajh lena.

## Verbs

| Verb | Kaun active hai | LLM mein |
|---|---|---|
| **SEND / RECV** | Dono side. Receiver pehle recv buffer post karta hai, sender usme SEND karta hai. | Chhote control messages. |
| **WRITE** | Sirf sender. Remote address + rkey pe seedha likhta hai. Remote CPU involved nahi. | Bade gradient chunks. NCCL ka common data path. |
| **READ** | Sirf sender. Remote se kheenchta hai. | Kam, jab pull chahiye. |
| **Atomic** | Fetch-and-add, compare-and-swap, NIC pe. | Locks, counters. Collectives ka bulk data nahi. |

WRITE **one-sided** hai. Remote application ko us packet ka callback nahi chahiye. NIC ne DMA kar diya, memory badal gayi. Sender ko completion CQ pe milta hai.

## Kernel bypass

TCP pe har `send()` syscall hai. Kernel buffer copy karta hai, TCP state machine chalata hai.

RDMA pe process apna QP **doorbell** MMIO se bajaata hai. Work request NIC ki queue mein chala jaata hai. Chhote message pe syscall nahi hoti. CPU payload ko read nahi karti. Isliye latency NIC + PCIe + switch hops hai, kernel scheduling nahi.

Yeh "CPU band hai" nahi hai. CPU work request banati hai aur CQ poll karti hai. Jo CPU nahi karti: payload copy, checksum, aur remote side pe `recv()`.

## GPUDirect RDMA

Abhi tak RDMA host DRAM ki baat thi. Gradient HBM mein hai. Bina **GPUDirect RDMA** ke NIC sirf host DRAM DMA kar sakta hai, isliye pehle `cudaMemcpy` HBM → DRAM, phir RDMA. PCIe do baar.

GPUDirect RDMA pe GPU ki memory bhi ek MR ban jaati hai. NIC ka DMA engine HBM tak pahunchta hai, GPU aur NIC ke beech peer mapping se. Kernel module purane stacks pe `nvidia-peermem` hai. Naye drivers mein yeh support driver ke andar hai.

```text
TCP
  HBM --PCIe-- DRAM --kernel-- NIC --wire-- NIC --kernel-- DRAM --PCIe-- HBM

GPUDirect RDMA
  HBM --DMA-- NIC --wire-- NIC --DMA-- HBM
```

Condition: NIC aur GPU **paas** hone chahiye. Same PCIe switch (**PIX**) best hai. CPU socket ke across (**SYS**) GPUDirect band ho sakta hai, kyunki DMA ko QPI/UPI cross karna padta hai. `nvidia-smi topo -m` yeh distance dikhata hai. Legend Day 12 pe hai.

## Latency aur bandwidth alag hain

Day 2 wala farq yahan bhi hai.

**Chhota message**, jaise koi sync flag. Time = PCIe + NIC + ek switch hop. Bandwidth matter nahi karti. 64 bytes 400 Gb/s pe nanoseconds ka wire time hai, lekin setup microseconds le sakta hai. Isse **latency-bound** kehte hain.

**Bada message**, 14 GB gradients. Time ≈ size / bandwidth. 400 Gb/s rail pe seconds ke fractions. CPU bypass ho chuka hai, ab limit optical port hai. Isse **bandwidth-bound** kehte hain.

NCCL dono ke liye alag protocol rakhta hai. Chhote ke liye **LL** (low latency), bade ke liye **Simple**. Beech mein **LL128**. Day 12 ke log mein yeh naam dikhenge. Protocol galat ho to chhota AllReduce bhi bandwidth wale overhead pe chal padta hai.

## Aaj ka rule

RDMA = NIC remote MR pe WRITE/READ karta hai, rkey ke saath, remote CPU ke `recv()` ke bina. GPUDirect wohi DMA HBM tak le jaata hai. TCP wala CPU copy is path mein nahi hai. Chhota message latency se atakta hai, bada gradient bandwidth se.
