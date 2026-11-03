# 📅 Day 11: NCCL (NVIDIA Collective Communications Library) Primitives

Aaj ke day ka core objective yeh samajhna hai ki **NCCL** multiple GPUs ke beech gradients aur tensors ko aapas me sync karne ke liye kaun-kaun se collective communication patterns (Primitives) run karta hai, unka mathematical flow kya hota hai, aur **Ring-AllReduce** algorithm bandwidth ko kaise optimize karta hai.

---

## 1. NCCL Collective Primitives (The Big 4)

Distributed LLM training me multi-GPU communication basically 4 main primitives par chalti hai:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       NCCL Collective Primitives                            │
├──────────────────┬──────────────────┬───────────────────┬───────────────────┤
│    AllReduce     │   ReduceScatter  │     AllGather     │     Broadcast     │
└──────────────────┴──────────────────┴───────────────────┴───────────────────┘
```

---

### A. AllReduce (Sum / Reduction)
* **Goal:** Har GPU ke paas local tensor (jaise local gradients) hota hai. Sabhi GPUs ke tensors ka **element-wise sum** karke *final reduced result sabhi GPUs ko wapas deliver* karna.
* **Use Case:** PyTorch DDP me backward pass ke baad gradients ko saare GPUs par ek sath sync karna.

```
INPUT:
  GPU 0: [ A ]
  GPU 1: [ B ]
  GPU 2: [ C ]
  GPU 3: [ D ]

OPERATION: AllReduce (SUM)
   │
   ▼
OUTPUT:
  GPU 0: [ A + B + C + D ]
  GPU 1: [ A + B + C + D ]
  GPU 2: [ A + B + C + D ]
  GPU 3: [ A + B + C + D ]
```

---

### B. ReduceScatter
* **Goal:** Sub-vectors ko reduce (sum) bhi karta hai, aur final outcome ko split karke har GPU ko uska dedicated single segment (scattered chunk) hi deta hai.
* **Use Case:** DeepSpeed **ZeRO-2** gradient partitioning me (har GPU sirf apne assigned parameters ke gradients ko reduce karke rakhta hai).

```
INPUT: (Each GPU holds 4 chunks)
  GPU 0: [ A0 | A1 | A2 | A3 ]
  GPU 1: [ B0 | B1 | B2 | B3 ]
  GPU 2: [ C0 | C1 | C2 | C3 ]
  GPU 3: [ D0 | D1 | D2 | D3 ]

OPERATION: ReduceScatter (SUM)
   │
   ▼
OUTPUT: (Each GPU gets 1 reduced chunk)
  GPU 0: [ A0 + B0 + C0 + D0 ]
  GPU 1: [ A1 + B1 + C1 + D1 ]
  GPU 2: [ A2 + B2 + C2 + D2 ]
  GPU 3: [ A3 + B3 + C3 + D3 ]
```

---

### C. AllGather
* **Goal:** Har GPU ke paas apna ek chhota tensor hota hai. Sabhi GPUs apne-apne tensors share karte hain taaki aakhri me sabke paas pura concatenated global tensor ho.
* **Use Case:** Tensor Parallelism (TP) aur DeepSpeed **ZeRO-3** parameter reconstruction (forward pass se pehle partitioned weights ko ek sath lana).

```
INPUT:
  GPU 0: [ A0 ]
  GPU 1: [ B1 ]
  GPU 2: [ C2 ]
  GPU 3: [ D3 ]

OPERATION: AllGather
   │
   ▼
OUTPUT:
  GPU 0: [ A0 | B1 | C2 | D3 ]
  GPU 1: [ A0 | B1 | C2 | D3 ]
  GPU 2: [ A0 | B1 | C2 | D3 ]
  GPU 3: [ A0 | B1 | C2 | D3 ]
```

> 💡 **Key Insight:** $\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$

---

### D. Broadcast
* **Goal:** System me ek Root GPU ke paas data hota hai, jise baaki saare GPUs par exact copy karke bhej diya jata hai.
* **Use Case:** Model weights initialization ya master config variables ko baaki ranks me update karna.

```
INPUT:
  GPU 0 (Root): [ DATA ]
  GPU 1:        [  --  ]
  GPU 2:        [  --  ]
  GPU 3:        [  --  ]

OPERATION: Broadcast (from Root=GPU 0)
   │
   ▼
OUTPUT:
  GPU 0: [ DATA ]
  GPU 1: [ DATA ]
  GPU 2: [ DATA ]
  GPU 3: [ DATA ]
```

---

## 2. Deep Dive: Ring-AllReduce Algorithm

Naively agar ek Root GPU sabhi GPUs se gradients collect karke sum kare aur fir sabko wapas bheje, toh Root GPU network bottleneck ban jayega ($O(N)$ network transfer). 

Is problem ko solve karne ke liye NCCL **Ring-AllReduce** implementation run karta hai, jisme GPUs logical ring network layout banate hain.

```
         ┌─────────────┐       NVLink / RDMA      ┌─────────────┐
         │    GPU 0    │ ───────────────────────► │    GPU 1    │
         └──────┬──────┘                          └──────┬──────┘
                ▲                                        │
  NVLink / RDMA │                                        │ NVLink / RDMA
                │                                        ▼
         ┌──────┴──────┐                          ┌─────────────┐
         │    GPU 3    │ ◄─────────────────────── │    GPU 2    │
         └─────────────┘       NVLink / RDMA      └─────────────┘
```

---

### Two-Step Execution Flow:

Assume $N$ GPUs hain (e.g., $N=4$). Entire data array ko $N$ equal chunks me break kar diya jata hai: `Chunk 0, Chunk 1, Chunk 2, Chunk 3`.

#### Step 1: Reduce-Scatter Phase ($N-1$ Steps)
* Har step me GPU $i$ apna chunk nearest successor GPU $(i+1) \pmod N$ ko bhejta hai.
* Receive karne wala GPU aane wale chunk me apna matching local chunk **add** kar leta hai.
* **$N-1$ steps ke baad:** Har GPU ke paas **exactly 1 chunk** ka fully reduced final sum hota hai.

```
[Initial]  GPU 0: [A0,A1,A2,A3]  GPU 1: [B0,B1,B2,B3]  GPU 2: [C0,C1,C2,C3]  GPU 3: [D0,D1,D2,D3]

Step 1:    GPU 0 sends A3 to GPU 1 | GPU 1 sends B0 to GPU 2 | GPU 2 sends C1 to GPU 3 ...
Step 2:    GPU 0 sends D2+C2 to GPU 1 ...

After N-1 Steps (Reduce-Scatter Done):
GPU 0 holds:  A0 + B0 + C0 + D0  (Fully Reduced Chunk 0)
GPU 1 holds:  A1 + B1 + C1 + D1  (Fully Reduced Chunk 1)
GPU 2 holds:  A2 + B2 + C2 + D2  (Fully Reduced Chunk 2)
GPU 3 holds:  A3 + B3 + C3 + D3  (Fully Reduced Chunk 3)
```

#### Step 2: All-Gather Phase ($N-1$ Steps)
* Ab fully reduced chunks ko pure ring me circulate kiya jata hai taaki baki 3 GPUs ke paas bhi baki missing chunks ki complete copy pahunch jaye.
* Is Phase me koi **addition nahi hota**, sirf direct value copies passing hoti hai.

```
After another N-1 Steps (All-Gather Done):
GPU 0: [Sum0, Sum1, Sum2, Sum3]
GPU 1: [Sum0, Sum1, Sum2, Sum3]
GPU 2: [Sum0, Sum1, Sum2, Sum3]
GPU 3: [Sum0, Sum1, Sum2, Sum3]
```

---

## 3. Mathematical Bandwidth Analysis

Ring-AllReduce algorithm ka sabse bada advantage yeh hai ki **Data Transfer Amount GPU Count ($N$) par lineally expand nahi hota**.

Let:
* $S$ = Total Size of Tensor (in Bytes)
* $N$ = Number of GPUs in the ring

### Calculation:

$$\text{Data sent per GPU in Reduce-Scatter} = (N - 1) \times \left( \frac{S}{N} \right)$$

$$\text{Data sent per GPU in All-Gather} = (N - 1) \times \left( \frac{S}{N} \right)$$

$$\text{Total Data Sent per GPU} = 2 \times \left( \frac{N - 1}{N} \right) \times S$$

If $N \to \infty$ (Bohot bada GPU Cluster):

$$\lim_{N \to \infty} 2 \times \left( \frac{N - 1}{N} \right) \times S \approx 2S$$

> 📌 **Takeaway:** System me kitne bhi GPUs add kar lo ($N=8$ ho ya $N=1024$), har GPU maximum **$2 \times \text{Data Size}$** ke barabar hi payload traffic transport karega! Network saturated nahi hoga.

---

## 4. PyTorch Distributed Python Code Example

`torch.distributed` Python primitives backend `nccl` ke saath direct mapping automate karta hai:

```python
import os
import torch
import torch.distributed as dist

def run_nccl_primitives_demo(rank, world_size):
    # Setup NCCL Process Group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Local Tensor Initialization
    data = torch.tensor([rank + 1.0, (rank + 1.0) * 10.0], device=f"cuda:{rank}")
    print(f"[Rank {rank}] Initial Tensor: {data.cpu().numpy()}")

    # 1. AllReduce Demo (Summing tensors across all GPUs)
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] After AllReduce (SUM): {data.cpu().numpy()}")

    # 2. AllGather Demo
    local_val = torch.tensor([rank * 100.0], device=f"cuda:{rank}")
    gathered_output = [torch.zeros(1, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(gathered_output, local_val)
    
    if rank == 0:
        res = [t.item() for t in gathered_output]
        print(f"[Rank 0] AllGather Result from all ranks: {res}")

    dist.destroy_process_group()
```

---

## 5. Day 11 Summary Checklist

1. **AllReduce:** Gradient Sync ka spine. $\text{ReduceScatter} + \text{AllGather}$ se milkar banta hai.
2. **ReduceScatter:** ZeRO-2 me gradients split karne ke liye mandatory.
3. **AllGather:** ZeRO-3 aur Tensor Parallelism (TP) me global tensors reconstruct karne ke liye required.
4. **Ring-AllReduce Bandwidth:** Transferred Volume per GPU fixed $\approx 2S$ hota hai regardless of cluster size.