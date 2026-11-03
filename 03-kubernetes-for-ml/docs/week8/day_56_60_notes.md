# 📅 Day 56-60: Distributed Inference & Speculative Decoding

Aapka yeh module **Large-Scale Distributed Inference & Advanced Acceleration Techniques** par focus karta hai. Jab model size itna bada ho jata hai ki ek single GPU ke VRAM mein fit nahi ho sakta (jaise 70B+ parameters), ya jab humein massive scale par low-latency token generation chahiye, tab hume distributed training ki tarah **Distributed Inference** ki zaroorat padti hai. Aaj hum Tensor Parallelism, Pipeline Parallelism at inference, aur **Speculative Decoding** ke core concepts ko samjhenge.

## 1. Distributed Inference: Tensor Parallelism (TP) vs. Pipeline Parallelism (PP)

Inference ke dauran, distributed execution training se thoda alag behave karta hai kyunki yahan backward pass (gradients) nahi hote, sirf forward pass aur KV cache synchronization hoti hai.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Tensor Parallelism vs. Pipeline Parallelism                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Tensor Parallelism (TP) within Node:                                     │
│    - Splits individual weight matrices (Attention heads, MLP layers) across   │
│      multiple GPUs connected via high-speed NVLink.                         │
│    - Ultra-low latency, but requires heavy intra-node communication (AllReduce).│
│                                                                             │
│ 2. Pipeline Parallelism (PP) across Nodes:                                  │
│    - Splits model layers sequentially across different nodes (e.g., Node 1:   │
│      Layers 1-16, Node 2: Layers 17-32).                                    │
│    - Reduces communication overhead, but introduces "bubble" idle time.     │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Key Considerations for Inference TP/PP:

* **TP Communication Overhead:** Megatron-style TP introduces two `All-Reduce` operations per Transformer layer. Without high-bandwidth NVLink/NVSwitch (e.g., 900 GB/s on H100), inter-GPU communication becomes a major bottleneck.

* **Pipeline Bubbles:** To maximize pipeline utilization during inference, request batching must be optimized using continuous batching so that intermediate stages remain busy.

## 2. Speculative Decoding: Accelerating Token Generation

As discussed earlier, the decode phase is memory-bandwidth bound. GPUs spend most of their time fetching heavy weights from HBM for just **one token at a time**. **Speculative Decoding** solves this by using a small, fast "Draft Model" to generate multiple candidate tokens, which are then verified in a single parallel forward pass by the large, slow "Target Model".

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Speculative Decoding Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Draft Model (e.g., 1B parameter model):                                  │
│    - Generates $K$ candidate tokens rapidly sequentially (e.g., 5 tokens).   │
│                                                                             │
│ 2. Target Model (e.g., 70B parameter model):                                │
│    - Performs ONE parallel forward pass over the prompt + 5 draft tokens.    │
│    - Verifies logits and accepts valid tokens based on probability checks.  │
│                                                                             │
│ Result: If 4 out of 5 tokens are accepted, the target model processes 5      │
│ tokens in the time it normally takes to generate ~1.2 tokens!               │
└─────────────────────────────────────────────────────────────────────────────┘

```

### The Acceptance Criterion (Sampling Algorithm):

* Let the draft model propose token $x_t$ with probability $q(x_t)$ and target model assign probability $p(x_t)$.

* A token is accepted if a random uniform draw $r \in [0, 1)$ satisfies:
$$\min\left(1, \frac{p(x_t)}{q(x_t)}\right) \ge r$$

* If a token is rejected, generation rolls back to the last accepted token, and a correction token is sampled from the adjusted target distribution.

## 3. Summary Checkpoints for Days 56-60

1. **Tensor Parallelism:** Splits layers across NVLink-connected GPUs to minimize latency for large models.

2. **Pipeline Parallelism:** Distributes sequential layers across nodes to fit massive models while managing pipeline bubbles.

3. **Speculative Decoding:** Uses a small draft model to generate candidate tokens and a target model to verify them in parallel, drastically reducing memory bandwidth bottlenecks and accelerating token generation.