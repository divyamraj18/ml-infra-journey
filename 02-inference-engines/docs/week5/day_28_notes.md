# 📅 Day 28: LLM Inference Bottlenecks & Modern Serving Optimizations

Aapka **Week 5: Storage Architectures & LLM Inference Bottlenecks** module yahan se shuru hota hai. Today’s objective is to understand why LLM inference behaves completely differently from training, how the **Prefill** (Compute-bound) and **Decode** (Memory-bound) phases function, the exact memory footprint of the **KV-Cache**, and how **PagedAttention**, **Continuous Batching**, and **Speculative Decoding** solve these bottlenecks.

---

## 1. Inference vs. Training: Data Flow & Bottleneck Shift

During training, backpropagation requires storing activations for every layer, leading to huge VRAM utilization. Communication (AllReduce/AllToAll) and matrix multiplications ($C = A \times B$) dominate performance.

Inference, however, works in a **Two-Phase Autoregressive Generation Loop**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LLM Inference Execution Flow                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
  1. Prefill Phase (Prompt Ingestion) ──► Input prompt matrix processed in PARALLEL.
                                          High Arithmetic Intensity → Compute-Bound.
                                       │
                                       ▼
  2. Decode Phase (Token Generation)  ──► Tokens generated ONE BY ONE.
                                          Low Arithmetic Intensity → Memory-Bound (HBM Latency).
```

---

## 2. Prefill vs. Decode Phase Deep Dive

### A. Prefill Phase (Prompt Phase)
* **Goal:** Process the user’s input prompt $N$ tokens at once to initialize the context.
* **Compute Characteristics:** Matrix-Matrix Multiplication (GEMM). Because all $N$ tokens are available concurrently, GPUs can maximize tensor core utilization.
* **Bottleneck:** **Compute-bound**. The GPU ALUs are saturated with floating-point operations.

### B. Decode Phase (Token Generation Phase)
* **Goal:** Generate output tokens autoregressively ($t_1, t_2, \dots, t_M$), feeding the newest token back into the model to predict the next token.
* **Compute Characteristics:** Matrix-Vector Multiplication (GEMV). For every new token, all model weights ($W$) and previous Key-Value pairs must be loaded from GPU HBM to SRAM for just *one* token calculation.
* **Arithmetic Intensity:** Extremely low ($\ll 1 \text{ FLOP/byte}$).
* **Bottleneck:** **Memory-Bound** (HBM Bandwidth latency).

```
===================================================================================
ARITHMETIC INTENSITY COMPARISON
===================================================================================
  Prefill:  FLOPs / Bytes Read = HIGH   ---> GPU Cores Saturated (Compute Bound)
  Decode:   FLOPs / Bytes Read = LOW    ---> GPU Cores Idle Waiting for Memory (Memory Bound)
===================================================================================
```

---

## 3. KV-Cache Memory Footprint & Calculation

To avoid recomputing Keys ($K$) and Values ($V$) for all preceding tokens at each decoding step, we cache them in VRAM: **KV-Cache**.

### Mathematical Formula for KV-Cache Size:

$$\text{KV-Cache Memory Per Token (Bytes)} = 2 \times 2 \times d_{\text{model}} \times N_{\text{layers}}$$

Where:
* $2$: Two matrices ($K$ and $V$).
* $2$: FP16 / BF16 precision ($2 \text{ bytes per element}$).
* $d_{\text{model}}$: Total hidden dimension (or $N_{\text{heads}} \times d_{\text{head}}$).
* $N_{\text{layers}}$: Number of Transformer layers.

### Total KV-Cache Size Formula for Batch Size $B$ & Context Length $L$:

$$\text{Total Memory (GB)} = \frac{2 \times 2 \times N_{\text{layers}} \times d_{\text{model}} \times L \times B}{10^9}$$

#### Concrete Example: Llama-3-70B
* $N_{\text{layers}} = 80$
* $d_{\text{model}} = 8192$ (Grouped Query Attention reduces KV heads, e.g., $N_{\text{kv\_heads}} = 8$, $d_{\text{head}} = 128 \Rightarrow d_{\text{kv}} = 1024$)
* Context Length ($L$) = $8192$ tokens
* Batch Size ($B$) = $32$

Using GQA ($d_{\text{kv}} = 1024$):
$$\text{Memory Per Token} = 2 \times 2 \times 1024 \times 80 = 327,680 \text{ bytes} \approx 0.327 \text{ MB/token}$$
$$\text{For } 8192 \text{ tokens/user} = 0.327 \text{ MB} \times 8192 \approx 2.68 \text{ GB per sequence}$$
$$\text{For Batch Size 32} = 32 \times 2.68 \text{ GB} \approx \mathbf{85.76 \text{ GB}}$$

> **Insight:** The KV-Cache alone for a batch of 32 requests can exceed the total memory of an 80GB A100 GPU!

---

## 4. PagedAttention Architecture (vLLM)

In standard PyTorch serving implementations, KV-caches are allocated as **contiguous memory blocks** for the maximum sequence length (e.g., 2048 or 8192). This leads to massive **Virtual Memory Fragmentation**:

1. **Internal Fragmentation:** Reserving memory for max context length that the user may never reach.
2. **External Fragmentation:** Dynamic allocation causes scattered free memory pockets.

### How PagedAttention (vLLM) Solves Memory Waste:

PagedAttention borrows the concept of **Virtual Memory Paging** from traditional Operating Systems (OS):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PagedAttention Architecture                        │
└─────────────────────────────────────────────────────────────────────────────┘

  Logical Key-Value Blocks (Per Request)
  ┌──────────┬──────────┬──────────┐
  │ Block 0  │ Block 1  │ Block 2  │
  └────┬─────┴────┬─────┴────┬─────┘
       │          │          │
       ▼          ▼          ▼ Block Table Mapping
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ Physical Memory Blocks (Non-Contiguous HBM Allocation)                    │
  │ ┌──────────┐      ┌──────────┐      ┌──────────┐                          │
  │ │ Block 12 │ ---> │ Block 3  │ ---> │ Block 89 │                          │
  │ └──────────┘      └──────────┘      └──────────┘                          │
  └───────────────────────────────────────────────────────────────────────────┘
```

* **Fixed-size Blocks:** Memory is split into small physical blocks (e.g., 16 tokens per block).
* **Block Tables:** Maintains a physical-to-logical mapping table.
* **Zero Fragmentation:** Memory is allocated dynamically on the fly as tokens generate. Waste is reduced to $<1\%$.

---

## 5. Continuous Batching (Iteration-Level Scheduling)

### Static Batching (Old Approach):
Processes waiting in a batch are tied together. If Request A generates 10 tokens and Request B generates 500 tokens, the GPU slots for Request A sit **idle** until Request B finishes.

```
STATIC BATCHING:
Req 1 (Short): |=== Token Gen ===|............... IDLE WAITING ..................|
Req 2 (Long) : |======================= Token Gen ===============================|
```

### Continuous Batching (Cell/Iteration-Level Batching):
Instead of batching at the *sequence level*, scheduling happens at the *iteration level*. 
* As soon as Request 1 finishes at step 10, it is evicted immediately.
* A new incoming request (Request 3) enters the batch at step 11 to fill the vacant slot.

```
CONTINUOUS BATCHING:
Req 1 (Short): |=== Token Gen ===|
Req 3 (New)  :                    |=========== Token Gen =================-------|
Req 2 (Long) : |======================= Token Gen ===============================|
```

---

## 6. Speculative Decoding

Speculative Decoding accelerates memory-bound **Decode Phase** generation without modifying the output target distribution.

### How it works:
1. **Draft Model (Small & Fast):** A lightweight model (e.g., Llama-68M) speculatively predicts $K$ candidate tokens (e.g., $K=5$) sequentially in high speed.
2. **Target Model (Large & Slow):** The large LLM (e.g., Llama-70B) runs a **single parallel forward pass** (Prefill-style GEMM) over all $K$ draft tokens concurrently.
3. **Verification Step:** Using modified rejection sampling, the target model accepts or rejects the draft tokens. 

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Speculative Decoding Execution                        │
└─────────────────────────────────────────────────────────────────────────────┘

  Draft Model (Fast):   Generates [T1, T2, T3, T4] sequentially (Low overhead)
                                      │
                                      ▼
  Target Model (Slow):  Validates [T1, T2, T3, T4] in ONE SINGLE compute step!
                        Result: Accept T1, T2, T3 | Reject T4
                                      │
                                      ▼
  Outcome: 3 tokens generated in 1 Target Model forward step instead of 3 steps!
```

---

## 7. Summary Checkpoints for Day 28

1. **Phase Bottlenecks:** Prefill is compute-bound (GEMM); Decode is memory-bound (GEMV).
2. **KV-Cache Inflation:** Caching Keys and Values scales linearly with sequence length and batch size, requiring optimization strategies like Grouped Query Attention (GQA).
3. **PagedAttention:** Solves memory fragmentation by borrowing virtual page mapping concepts from operating systems.
4. **Continuous Batching:** Schedules requests dynamically at iteration steps rather than sequence completion boundaries.
5. **Speculative Decoding:** Converts single-token memory-bound decoding steps into parallel compute-bound verification passes using a smaller draft model.