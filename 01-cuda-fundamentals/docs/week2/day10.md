# Day 10: RDMA

Day 9 ne kaha gradient CPU ke TCP stack se nahi guzarna chahiye. **RDMA (Remote Direct Memory Access)** wohi mechanism hai.

Socho do warehouses hain. TCP mein parcel pehle post office jaata hai, wahan koi use kholta hai, phir doosre warehouse ke counter pe `recv()` karta hai. RDMA mein truck seedha doosre warehouse ki shelf pe box rakh deta hai. Doosri taraf koi counter pe khada nahi hota.

NIC remote memory mein likhta hai. Remote CPU `recv()` nahi chalata.

## Teen cheezein jo NIC ko chahiye

Har side pe ek **RNIC** hota hai. DGX pe yeh ConnectX hai. RNIC tabhi DMA kar sakta hai jab use pages ka physical address pata ho, aur kernel ne woh mapping de di ho.

| Object | Simple matlab | Technical naam |
|---|---|---|
| Connection ka endpoint | Do queues: bhejne wali aur receive wali. TCP socket jaisa, lekin payload kernel mein nahi jaata. | **Queue Pair (QP)** |
| "Kaam khatam" ki list | NIC yahan likhta hai ki kaun si request ho gayi. CPU poll karta hai. Interrupt zaroori nahi. | **Completion Queue (CQ)** |
| Locked pages | NIC ke liye pin ki hui memory. Local key **lkey**. Remote writer ki key **rkey**. Bina rkey remote NIC tumhari memory nahi chhu sakta. | **Memory Region (MR)** |

**Protection Domain** in keys ko ek group mein band karta hai, taaki doosre process ka QP tumhara MR use na kar le.

Setup pe CPU ab bhi kaam karti hai: pages pin karo, QP banao, remote ko rkey aur address bhejo. Yeh control messages chhote hote hain. **Bootstrap** aksar TCP se hota hai. Uske baad ka data path RDMA hai.

Bootstrap ko data path mat samajhna. Log mein ek TCP line sirf "humne ek doosre ko address bataya" ho sakti hai. Gradient us line pe nahi ja raha.

## Verbs: truck kya kare

| Verb | Kaun active hai | LLM job mein |
|---|---|---|
| **SEND / RECV** | Dono side. Receiver pehle shelf khali karke bataata hai. Sender usme SEND karta hai. | Chhote control messages. |
| **WRITE** | Sirf sender. Remote address + rkey pe seedha likhta hai. Remote CPU involved nahi. | Bade gradient chunks. NCCL ka common data path. |
| **READ** | Sirf sender. Remote shelf se kheenchta hai. | Kam. Jab sender pull karna chahe. |
| **Atomic** | NIC pe fetch-and-add ya compare-and-swap. | Locks, counters. Collective ka bulk data nahi. |

WRITE **one-sided** hai. Remote application ko us packet ka callback nahi milta. NIC ne DMA kar diya, memory badal gayi. Sender ko completion apni CQ pe milti hai.

## Kernel bypass

TCP pe har `send()` ek syscall hai. Kernel buffer copy karta hai aur TCP state machine chalata hai.

RDMA pe process apne QP ka **doorbell** MMIO se bajata hai. Work request NIC ki queue mein chali jaati hai. Chhote message pe syscall nahi hoti. CPU payload ko read nahi karti. Latency NIC + PCIe + switch hops hai, kernel scheduling nahi.

Iska matlab CPU band hai, yeh nahi. CPU work request banati hai aur CQ poll karti hai. CPU yeh nahi karti: payload copy, checksum, aur remote side pe `recv()`.

## GPUDirect RDMA

Abhi tak truck CPU ke warehouse (DRAM) ki baat kar raha tha. Gradient GPU ke warehouse (HBM) mein hai.

Bina **GPUDirect RDMA** ke NIC sirf host DRAM DMA kar sakta hai. Pehle `cudaMemcpy` HBM se DRAM, phir RDMA. PCIe do baar.

GPUDirect pe GPU ki memory bhi ek MR ban jaati hai. NIC ka DMA engine HBM tak pahunchta hai, GPU aur NIC ke beech peer mapping se. Purane stacks pe kernel module `nvidia-peermem` hai. Naye drivers mein yeh support driver ke andar hai.

```text
TCP
  HBM --PCIe-- DRAM --kernel-- NIC --wire-- NIC --kernel-- DRAM --PCIe-- HBM

GPUDirect RDMA
  HBM --DMA-- NIC --wire-- NIC --DMA-- HBM
```

Condition: NIC aur GPU **paas** hone chahiye. Same PCIe switch, jise **PIX** kehte hain, best case hai. Do CPU sockets ke across, **SYS**, GPUDirect band ho sakta hai, kyunki DMA ko QPI ya UPI cross karna padta hai. `nvidia-smi topo -m` yeh distance dikhata hai. Legend Day 12 pe hai.

## Chhota message aur bada message

Day 2 wala farq yahan bhi hai.

**Chhota message**, jaise ek sync flag. Time = PCIe + NIC + ek switch hop. Bandwidth matter nahi karti. 64 bytes, 400 Gb/s pe, wire pe nanoseconds hain. Setup microseconds le sakta hai. Yeh **latency-bound** hai.

**Bada message**, 14 GB gradients. Time ≈ size / bandwidth. 400 Gb/s rail pe yeh second ka hissa hai. CPU path se hat chuka hai. Limit optical port hai. Yeh **bandwidth-bound** hai.

NCCL inke liye alag protocol rakhta hai.

| Protocol | Kab |
|---|---|
| **LL** (low latency) | Chhote messages |
| **LL128** | Beech ka size |
| **Simple** | Bade messages, jaise poora gradient bucket |

Yeh naam Day 12 ke log mein dikhte hain. Galat protocol ho to chhota AllReduce bhi bade-message wala overhead bharta hai.

## Aaj ka rule

RDMA = NIC remote MR pe WRITE ya READ karta hai, rkey ke saath, remote CPU ke `recv()` ke bina. GPUDirect wohi DMA HBM tak le jaata hai. TCP wali CPU copy is path pe nahi hai. Chhota message latency se atakta hai. Bada gradient bandwidth se.
