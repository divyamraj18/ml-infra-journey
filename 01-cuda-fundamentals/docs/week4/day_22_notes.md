# 📅 Day 22: Tensor Parallelism (TP) — Column & Row Parallel Layers

Aaj ke din se hum **Week 4 (Tensor Parallelism & Memory Optimization)** shuru kar rahe hain. Tensor Parallelism ka main motive yeh hai ki jab ek single Transformer Layer (Linear / Matrix Multiplication) itna bada ho jaye ki GPU memory me fit na aaye, tab hum weight matrix ko hi split (shard) karke alag-alag GPUs par compute karte hain.

---

## 1. Pipeline Parallelism vs. Tensor Parallelism

* **Pipeline Parallelism (PP):** Layers ko sequentially alag-alag GPUs par baant’ta hai (jaise GPU 0 ko Layers 1-12, GPU 1 ko Layers 13-24). Communication micro-batches ke activations ka hota hai (Inter-layer).
* **Tensor Parallelism (TP):** Single Layer ke weight matrices (jaise Multi-Head Attention ya MLP) ko intra-layer level par GPUs me split karta hai. Communication har forward aur backward pass ke dauran immediate hota hai (using NCCL AllReduce / AllGather).

---

## 2. Megatron-LM Style Tensor Parallelism (Shoey & Korthik)

NVIDIA ke famous Megatron-LM paper me Transformer Block ko efficiently parallelize karne ke liye do primary building blocks diye gaye hain:
1. **Column Parallel Linear Layer**
2. **Row Parallel Linear Layer**

---

## 3. Column Parallel Linear Layer

### Concept:
Column Parallel Layer me hum weight matrix $W$ ko **Columns** ke along divide (shard) karte hain.

Given: $Y = X \times W$
* Input $X$ saare GPUs par complete copy hota hai.
* Weight $W$ ko vertical split karke $W_1, W_2, \dots, W_k$ banaya jata hai.
* Har GPU local matrix multiplication karta hai: $Y_i = X \times W_i$

### Visual Representation (Column Parallel)

```text
Input Tensor X (Replicated on all GPUs)
┌────────────────────────┐
│         X              │
└────────────────────────┘
            │
            ├───────────────────────────────────────┐
            ▼                                       ▼
    [ GPU 0 Rank 0 ]                        [ GPU 1 Rank 1 ]
┌────────────────────────┐              ┌────────────────────────┐
│         X              │              │         X              │
└────────────────────────┘              └────────────────────────┘
            │                                       │
            │ × (MatMul)                            │ × (MatMul)
            ▼                                       ▼
┌────────────────────────┐              ┌────────────────────────┐
│        W_0             │              │        W_1             │
│   (Columns 0..N/2)     │              │   (Columns N/2..N)     │
└────────────────────────┘              └────────────────────────┘
            │                                       │
            ▼                                       ▼
┌────────────────────────┐              ┌────────────────────────┐
│        Y_0             │              │        Y_1             │
│   (Local Output)       │              │   (Local Output)       │
└────────────────────────┘              └────────────────────────┘
            │                                       │
            └───────────────────┬───────────────────┘
                                │
                                ▼
                     AllGather / Concatenate
                                │
                                ▼
                    Final Output Y = [Y_0 | Y_1]
```

### Communication Pattern:
* **Forward Pass:** Complete output $Y$ paane ke liye $Y_0$ aur $Y_1$ ko **AllGather** (concatenate) kiya jata hai.
* **Backward Pass:** Backward pass me gradient $\frac{\partial L}{\partial X}$ accumulate karne ke liye **AllReduce (Sum)** operation chalta hai.

---

## 4. Row Parallel Linear Layer

### Concept:
Row Parallel Layer me hum weight matrix $W$ ko **Rows** ke along divide karte hain.

Given: $Y = X \times W$
* Weight matrix $W$ ko horizontal split ($W_1, W_2$) kiya jata hai.
* Iske matching ke liye Input $X$ ko bhi Column-wise partition karna padta hai ($X_0, X_1$).
* Local MatMul output: $Y_0 = X_0 \times W_0$ aur $Y_1 = X_1 \times W_1$

### Visual Representation (Row Parallel)

```text
GPU 0 Input (X_0)                      GPU 1 Input (X_1)
┌──────────────────┐                   ┌──────────────────┐
│       X_0        │                   │       X_1        │
└──────────────────┘                   └──────────────────┘
         │                                      │
         │ × (MatMul)                           │ × (MatMul)
         ▼                                      ▼
┌──────────────────┐                   ┌──────────────────┐
│       W_0        │                   │       W_1        │
│  (Rows 0..K/2)   │                   │  (Rows K/2..K)   │
└──────────────────┘                   └──────────────────┘
         │                                      │
         ▼                                      ▼
┌──────────────────┐                   ┌──────────────────┐
│       Y_0        │                   │       Y_1        │
│  (Partial Output)│                   │  (Partial Output)│
└──────────────────┘                   └──────────────────┘
         │                                      │
         └──────────────────┬───────────────────┘
                            │
                            ▼
                     NCCL AllReduce (SUM)
                            │
                            ▼
                 Final Output Y = Y_0 + Y_1
```

### Communication Pattern:
* **Forward Pass:** Outputs $Y_0$ aur $Y_1$ ka sum leni ke liye **AllReduce (Sum)** kiya jata hai.
* **Backward Pass:** Backward pass me local gradients auto-split rehte hain (Identity pass-through), isliye backward me network communication ki zaroorat nahi hoti (Zero comm)!

---

## 5. MLP Block Combine: Column Parallel + Row Parallel (Megatron Trick)

Megatron-LM MLP block me Sequential Linear layers use hote hain:
$$\text{Output} = \text{Dropout}(\text{GeLU}(X \cdot W_1) \cdot W_2)$$

Agar hum traditional tareeqe se karein toh har layer ke baad communication hoga. Megatron ne clever pattern nikala:
1. $W_1$ (First Linear Layer) ko **Column Parallel** rakha.
2. GeLU Activation local outputs par perform kiya.
3. $W_2$ (Second Linear Layer) ko **Row Parallel** rakha.

### Benefit of Combining:
* Column Parallel ka output direct Row Parallel ka input ban jata hai!
* Beech me se **AllGather eliminate** ho jata hai!
* Pure MLP block me **sirf 1 AllReduce** forward me aur **1 AllReduce** backward me lagta hai.

```text
Input X ──► [ Column Parallel W1 ] ──► GeLU ──► [ Row Parallel W2 ] ──► AllReduce ──► Output
               (No Network Comm)                   (Only 1 AllReduce Here)
```

---

## 6. PyTorch Toy Code Implementation

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, process_group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pg = process_group
        self.world_size = dist.get_world_size(pg)
        self.rank = dist.get_rank(pg)
        
        # Partition out_features across GPUs
        assert out_features % self.world_size == 0
        self.out_features_per_partition = out_features // self.world_size
        
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))

    def forward(self, x):
        # Local matrix multiplication: Y_local = X @ W_local^T + b_local
        local_output = nn.functional.linear(x, self.weight, self.bias)
        
        # Gathering outputs across all TP ranks
        output_list = [torch.empty_like(local_output) for _ in range(self.world_size)]
        dist.all_gather(output_list, local_output, group=self.pg)
        
        # Concatenate along column dimension
        output = torch.cat(output_list, dim=-1)
        return output


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, process_group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pg = process_group
        self.world_size = dist.get_world_size(pg)
        self.rank = dist.get_rank(pg)
        
        # Partition in_features across GPUs
        assert in_features % self.world_size == 0
        self.in_features_per_partition = in_features // self.world_size
        
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

    def forward(self, x_partition):
        # Local matrix multiplication
        local_output = nn.functional.linear(x_partition, self.weight)
        
        # AllReduce Sum to aggregate partial results
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM, group=self.pg)
        return local_output
```

---

## 7. Summary Table

| Layer Type | Weight Split Dimension | Input Requirement | Forward Comm Primitive | Backward Comm Primitive |
| :--- | :--- | :--- | :--- | :--- |
| **Column Parallel** | Columns (`out_features`) | Full / Replicated | AllGather | AllReduce |
| **Row Parallel** | Rows (`in_features`) | Split / Partitioned | AllReduce | None (Identity) |
| **MLP Block (Merged)** | Col W1 + Row W2 | Full Input | **Single AllReduce** | **Single AllReduce** |
