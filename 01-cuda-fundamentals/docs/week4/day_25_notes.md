# 📅 Day 25: DeepSpeed ZeRO (Zero Redundancy Optimizer) Architecture

Aaj ke day ka core objective yeh samajhna hai ki LLM training ke dauran GPU Memory (HBM) me data kahan consume hota hai, Standard Data Parallelism me redundancy kyun paida hoti hai, aur **DeepSpeed ZeRO (ZeRO-1, ZeRO-2, ZeRO-3)** is redundancy ko complete strip out karke billion-parameter models ko train karna kaise possible banata hai.

---

## 1. Where Does GPU Memory Go During LLM Training?

Jab hum Mixed-Precision (FP16/BF16) me AdamW Optimizer ke sath model train karte hain, toh GPU Memory 2 main categories me bati hoti hai:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GPU HBM Memory Layout                            │
├──────────────────────────────────────────┬──────────────────────────────────┤
│        Model Memory (Static)             │     Residual Memory (Dynamic)    │
├──────────────────────────────────────────┼──────────────────────────────────┤
│ 1. FP16/BF16 Parameters                  │ 1. Activations (Forward Pass)    │
│ 2. FP16/BF16 Gradients                   │ 2. Temporary CUDA Buffers        │
│ 3. FP32 Master Parameters                │ 3. Memory Fragmentation          │
│ 4. FP32 Optimizer States (Adam Momentum  │                                  │
│    + Adam Variance)                      │                                  │
└──────────────────────────────────────────┴──────────────────────────────────┘
```

### The 16x Memory Rule (Mixed-Precision AdamW)

Agar kisi model ke **$P$ parameters** hain, toh memory footprint kitna hoga?

1. **Parameters (FP16/BF16):** $2P$ bytes
2. **Gradients (FP16/BF16):** $2P$ bytes
3. **Optimizer States (AdamW in FP32):**
   * FP32 Master Parameters: $4P$ bytes
   * FP32 Momentum (First Moment): $4P$ bytes
   * FP32 Variance (Second Moment): $4P$ bytes
   * **Total Optimizer Memory:** $12P$ bytes

$$\text{Total Static Model Memory} = 2P + 2P + 12P = 16P \text{ bytes}$$

> **Example:** Ek **7B Parameter model** ko train karne ke liye static memory = $16 \times 7 \text{ GB} = 112 \text{ GB}$ chahiye! Single 80GB H100 GPU par ye standard DDP me fit hi nahi ho sakta.

---

## 2. Standard DDP Memory Redundancy Problem

Standard Data Parallelism (DDP) me har GPU pure model parameters, gradients, aur optimizer states ki **duplicate copy** rakhta hai. 

```
GPU 0: [ Params (16x) ] [ Grads (16x) ] [ Opt States (16x) ] ── (100% Redundant!)
GPU 1: [ Params (16x) ] [ Grads (16x) ] [ Opt States (16x) ] ── (100% Redundant!)
GPU 2: [ Params (16x) ] [ Grads (16x) ] [ Opt States (16x) ] ── (100% Redundant!)
GPU 3: [ Params (16x) ] [ Grads (16x) ] [ Opt States (16x) ] ── (100% Redundant!)
```

Is exact redundancy ko remove karne ke liye Microsoft ne **ZeRO (Zero Redundancy Optimizer)** design kiya.

---

## 3. DeepSpeed ZeRO Stages Breakdown

ZeRO memory redundancy ko 3 incremental stages me eliminate karta hai:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ZeRO Stages Overview                              │
├─────────────────┬───────────────────────────────────┬───────────────────────┤
│ Stage           │ What is Partitioned?              │ Memory Reduction      │
├─────────────────┼───────────────────────────────────┼───────────────────────┤
│ **ZeRO-1 ($P_{os}$)**│ Optimizer States Only             │ 4x Reduction          │
│ **ZeRO-2 ($P_{os+g}$)**│ Optimizer States + Gradients      │ 8x Reduction          │
│ **ZeRO-3 ($P_{os+g+p}$)**│ Optimizer States + Grads + Params│ Linear with $N_{gpus}$│
└─────────────────┴───────────────────────────────────┴───────────────────────┘
```

---

### A. ZeRO Stage 1: Optimizer State Partitioning ($P_{os}$)

* **Mechanism:** Model parameters aur gradients FP16 me replicated rehte hain, lekin FP32 Optimizer States ($12P$ bytes) ko $N$ GPUs ke beech equal split (partition) kar diya jata hai.
* **Memory Saved:** Har GPU par optimizer memory $12P / N$ ho jaati hai.
* **Communication:** Forward/Backward pass local hota hai. Backward ke baad `ReduceScatter` se gradients respective optimizer partitions tak pahunchte hain, local updates ke baad `AllGather` se updated FP16 parameters broadcast hote hain.

```
=================================================================================
ZeRO-1 MEMORY LAYOUT (4 GPUs)
=================================================================================
GPU 0: [ FP16 Params ] [ FP16 Grads ] [ Opt State Chunk 0 ]
GPU 1: [ FP16 Params ] [ FP16 Grads ] [ Opt State Chunk 1 ]
GPU 2: [ FP16 Params ] [ FP16 Grads ] [ Opt State Chunk 2 ]
GPU 3: [ FP16 Params ] [ FP16 Grads ] [ Opt State Chunk 3 ]
```

---

### B. ZeRO Stage 2: Gradient Partitioning ($P_{os+g}$)

* **Mechanism:** Optimizer States ke sath-sath Gradients ($2P$ bytes) ko bhi GPUs ke beech partition kar diya jata hai.
* **How it works:** Backward pass ke dauran, jaise hi layer ka gradient compute hota hai, direct `ReduceScatter` execute hota hai. GPU sirf us layer ka gradient store karta hai jiska optimizer state uske paas hai!
* **Memory Saved:** Memory footprint $\approx 2P + 2P + \frac{12P + 2P}{N}$ bytes ho jata hai.

```
=================================================================================
ZeRO-2 MEMORY LAYOUT (4 GPUs)
=================================================================================
GPU 0: [ FP16 Params ] [ Grad Chunk 0 ] [ Opt State Chunk 0 ]
GPU 1: [ FP16 Params ] [ Grad Chunk 1 ] [ Opt State Chunk 1 ]
GPU 2: [ FP16 Params ] [ Grad Chunk 2 ] [ Opt State Chunk 2 ]
GPU 3: [ FP16 Params ] [ Grad Chunk 3 ] [ Opt State Chunk 3 ]
```

---

### C. ZeRO Stage 3: Parameter Partitioning ($P_{os+g+p}$)

* **Mechanism:** Parameters ($2P$ bytes) ko bhi sharded kar diya jata hai. Kisi bhi GPU me pure model parameters ka persistent store nahi hota!
* **Forward Pass Execution:** 
  1. Forward computation se pehle, GPUs `AllGather` karke current layer ke parameters assemble karte hain.
  2. Computation complete hote hi extra parameters memory se **free (drop)** kar diye jaate hain!
* **Backward Pass Execution:** 
  1. Gradient computation ke liye dubara `AllGather` se layer parameters gather hote hain.
  2. Execution ke baad immediately parameters & gradients free kar diye jaate hain.
* **Memory Saved:** Total memory footprint scales linearly with GPU count:

$$\text{Memory per GPU} \approx \frac{16P}{N_{gpus}}$$

```
=================================================================================
ZeRO-3 MEMORY LAYOUT (4 GPUs)
=================================================================================
GPU 0: [ Param Chunk 0 ] [ Grad Chunk 0 ] [ Opt State Chunk 0 ]
GPU 1: [ Param Chunk 1 ] [ Grad Chunk 1 ] [ Opt State Chunk 1 ]
GPU 2: [ Param Chunk 2 ] [ Grad Chunk 2 ] [ Opt State Chunk 2 ]
GPU 3: [ Param Chunk 3 ] [ Grad Chunk 3 ] [ Opt State Chunk 3 ]

* Layers dynamically gathered via AllGather on-the-fly and freed immediately *
```

---

## 4. Communication Overhead Comparison

ZeRO-1 aur ZeRO-2 me standard DDP jitna hi communication volume hota hai ($2P$ total volume transfer). 

ZeRO-3 me parameters ko compute time par fetch karna padta hai, isliye **1.5x communication overhead** badhta hai:

```
Standard DDP / ZeRO-1 / ZeRO-2:
Total Volume transferred = 2P  (via AllReduce or ReduceScatter + AllGather)

ZeRO-3:
Forward Pass AllGather    = 1P
Backward Pass AllGather   = 1P
Backward Pass ReduceScatter = 1P
Total Volume transferred  = 3P  (1.5x of Standard DDP)
```

---

## 5. Summary Checkpoints for Day 25

1. **16x Rule:** Mixed-precision AdamW training requires $16P$ bytes of static HBM memory.
2. **ZeRO-1 & ZeRO-2:** Partition optimizer states and gradients with **zero extra communication overhead** compared to standard DDP.
3. **ZeRO-3:** Partitions model parameters as well, enabling trillion-parameter model training by sacrificing ~50% extra communication bandwidth for on-the-fly parameter gathering.