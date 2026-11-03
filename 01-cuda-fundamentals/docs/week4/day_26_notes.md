# 📅 Day 26: ZeRO-Offload, ZeRO-Infinity & 3D Parallelism (TP + PP + DP/ZeRO)

Aaj ke day ka target yeh samajhna hai ki jab models GPU VRAM ke capacity se bhi bada ho jata hai, tab **ZeRO-Offload / ZeRO-Infinity** CPU/NVMe memory ko kaise leverage karte hain, aur 100B+ parameter scale par **3D Parallelism (Tensor Parallelism + Pipeline Parallelism + Data/ZeRO Parallelism)** ko ek sath kaise organize aur stack kiya jata hai.

---

## 1. ZeRO-Offload & ZeRO-Infinity (Pushing Limits Beyond GPU VRAM)

ZeRO-3 local GPU memory overhead ko drastically reduce kar deta hai, par agar hum multi-billion parameter models limited GPUs par train kar rahe hon, tab bhi memory fill ho sakti hai. DeepSpeed ne is bottleneck ko handle karne ke liye CPU RAM aur NVMe SSDs ko memory hierarchy me incorporate kiya.

### A. ZeRO-Offload (CPU Memory Offloading)

Standard Mixed-Precision AdamW training me memory usage ka sabse bada chunk **Optimizer States (FP32 Master Weights, FP32 Momentum, FP32 Variance)** aur **FP32 Gradients** lete hain.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ZeRO-Offload Memory Split                            │
├──────────────────────────────────────┬──────────────────────────────────────┤
│              GPU Memory              │         Host CPU RAM / Compute       │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • FP16/BF16 Model Parameters         │ • FP32 Optimizer States              │
│ • Activations (Forward/Backward)     │ • FP32 Gradients (Pushed from GPU)   │
│ • Local FP16 Forward/Backward Kernels│ • CPU AdamW Optimizer Execution Step │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

#### Execution Cycle of ZeRO-Offload:
1. **GPU Forward & Backward Pass:** GPU FP16 weights ke zariye fast forward aur backward computation finish karta hai.
2. **Gradient Offload to CPU:** Computation ke baad FP16 gradients ko PCIe bus ke through Host CPU RAM me stream-offload kar diya jata hai.
3. **CPU Optimizer Step:** CPU compute cores (AVX/AMX instruction set) CPU RAM me majood FP32 Master Weights aur Optimizer States par AdamW Update rule apply karte hain.
4. **Updated FP16 Weights Stream Back:** Naye updated FP16 parameters ko Wapas GPU HBM me transfer kar diya jata hai next iteration ke forward pass ke liye.

---

### B. ZeRO-Infinity (NVMe Storage Extension)

Jab model weights aur states CPU RAM ki capacity ko bhi exceed kar jayein, tab **ZeRO-Infinity** NVMe SSDs ko unified virtual memory mapping layer me join karta hai.

* **NVMe Direct Offloading:** Micro-batch parameter transfers GPU Memory $\leftrightarrow$ CPU Memory $\leftrightarrow$ NVMe SSDs ke beech parallel asynchronous channels me pipeline hote hain.
* **Overlap via Prefetching:** Agle transformer layer ke required FP16 weights launch hone se pehle NVMe/CPU se pehle hi GPU SRAM/HBM me pre-fetch kar liye jaate hain taaki GPU idle na baithe.

---

## 2. 3D Parallelism Architecture (TP + PP + DP/ZeRO)

Extreme Scale Models (e.g., Llama-3 70B/405B, Megatron-Turing 530B) training me single parallelism technique (sirf DDP, sirf TP ya sirf PP) efficient nahi rehti. Production scale par teeno parallelisms ko combine kiya jata hai jise **3D Parallelism** kehte hain.

```
                             ┌───────────────────────────────────┐
                             │    Global Cluster (e.g. 64 GPUs)   │
                             └─────────────────┬─────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
    ┌─────────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
    │ Tensor Parallel (TP)│         │Pipeline Parallel(PP)│         │  Data Parallel (DP) │
    ├─────────────────────┤         ├─────────────────────┤         ├─────────────────────┤
    │ Intra-Node (NVLink) │         │ Inter-Node (NVSwitch│         │ Inter-Node Network  │
    │ High Bandwidth Splitting     │ Layer-wise sequential│         │ Ring-AllReduce /    │
    │ inside MHA/MLP      │         │ partitioning        │         │ ZeRO partitioning   │
    └─────────────────────┘         └─────────────────────┘         └─────────────────────┘
```

---

## 3. How 3D Parallel Groups Are Structured

Maan lijiye hamare paas **64 GPUs** ka ek multi-node GPU cluster hai. Hum 3D parallelism dimensions configure karte hain:

* **Tensor Parallel Degree ($TP$):** 4
* **Pipeline Parallel Degree ($PP$):** 4
* **Data Parallel Degree ($DP$):** 4

$$\text{Total GPUs} = TP \times PP \times DP = 4 \times 4 \times 4 = 64 \text{ GPUs}$$

### Communication Priority Rules:

1. **Tensor Parallelism (TP = 4):** Ek hi Node ke 4 NVLink-connected GPUs par mapped hota hai. Kyunki intra-layer AllReduce/AllGather ka frequency micro-step level par intense hota hai, use fast NVLink interconnect limit me rakha jata hai.
2. **Pipeline Parallelism (PP = 4):** Nodes ke across sequential layers (e.g., Node 1 $\rightarrow$ Node 2 $\rightarrow$ Node 3 $\rightarrow$ Node 4) split hota hai. Low-frequency Point-to-Point (P2P) InfiniBand RDMA transfers use karta hai.
3. **Data Parallelism (DP/ZeRO = 4):** Identical pipeline stage replicas ke beech gradients sync/partition karne ke liye run hota hai.

---

## 4. Communication & Memory Efficiency Comparison Matrix

| Strategy | Memory Saving per GPU | Network Communication Overhead | Primary Interconnect Dependency |
| :--- | :--- | :--- | :--- |
| **Standard DDP** | $1\times$ (No memory saving, full replica) | Moderate (AllReduce gradients per step) | NVLink / InfiniBand |
| **ZeRO-1 / 2 / 3** | Up to $1/8\times$ (ZeRO-3 splits weights & states) | Low to Medium (AllGather + ReduceScatter) | InfiniBand / RoCE |
| **Tensor Parallelism (TP)** | $1/TP\times$ Layer Memory | Extremely High (AllReduce inside Forward & Backward) | NVLink Only (Intra-Node) |
| **Pipeline Parallelism (PP)** | $1/PP\times$ Layer Memory | Lowest (P2P micro-batch activation passing) | PCIe / InfiniBand |
| **ZeRO-Offload** | Highest (VRAM pressure shifts to RAM/NVMe) | High (PCIe bus bound transfer rate) | PCIe Gen4/Gen5 |

---

## 5. Day 26 Summary Checklist

1. **ZeRO-Offload:** CPU RAM aur CPU processing cores par FP32 Optimizer States aur update compute offload karke VRAM save karta hai.
2. **ZeRO-Infinity:** Large scale model training capability extended to NVMe SSD storage through dynamic bandwidth pre-fetching.
3. **3D Parallelism Gold Standard:** Modern LLMs scale by stacking **TP** inside nodes (NVLink), **PP** across nodes (Sequential layers), and **DP/ZeRO** across world replicas for maximum compute efficiency and zero bandwidth bottlenecks.