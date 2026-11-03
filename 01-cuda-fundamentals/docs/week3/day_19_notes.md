# 📅 Day 19: Pipeline Parallelism — Bubble Problem & Minimization Strategies

Aaj ke day ka primary goal hai **Pipeline Bubble** ko mathematically calculate karna, schedule variations (GPipe vs 1F1B vs Interleaved 1F1B) ke impact ko samajhna, aur Memory Overhead vs Bubble Fraction ke trade-offs ko master karna.

---

## 1. Mathematical Breakdown of Pipeline Bubble

Pipeline Parallelism me **Bubble Fraction ($t_{\text{bubble}}$)** wo proportion hai jo GPUs output processing ka wait karte waqt idle guzarte hain.

### Key Variables:
* $p$: Number of Pipeline Stages (Number of GPUs in the pipeline)
* $m$: Number of Micro-batches in a mini-batch
* $t_f$: Time taken for one forward pass of a micro-batch on a single stage
* $t_b$: Time taken for one backward pass of a micro-batch on a single stage (typically $t_b \approx 2 \times t_f$)

### A. Non-Interleaved GPipe Schedule (Naive Micro-batching)

GPipe me sabhi micro-batches ke forward passes pehle run hote hain, aur uske baad saare backward passes.

$$\text{Total Active Time} = m \cdot (t_f + t_b)$$
$$\text{Total Idle Time per GPU} = (p - 1) \cdot (t_f + t_b)$$

$$\text{Bubble Fraction } (F_{\text{bubble}}) = \frac{(p - 1) \cdot (t_f + t_b)}{m \cdot (t_f + t_b) + (p - 1) \cdot (t_f + t_b)} = \frac{p - 1}{m + p - 1}$$

> **Key Takeaway:** Agar $m \gg p$ (Micro-batches is much larger than stages), toh bubble fraction $\approx \frac{p-1}{m}$ ho jata hai. Par high $m$ ka matlab hai heavy activation storage in memory!

---

## 2. Schedule Comparison: GPipe vs Standard 1F1B vs Interleaved 1F1B

### Visual Comparison of Pipeline Schedules

```
1. GPipe (High Memory Usage, Large Bubble)
   GPU 0: [F1][F2][F3][F4]-------------------------[B4][B3][B2][B1]
   GPU 1: ----[F1][F2][F3][F4]-----------------[B4][B3][B2][B1]----
   GPU 2: --------[F1][F2][F3][F4]---------[B4][B3][B2][B1]--------
   GPU 3: ------------[F1][F2][F3][F4][B4][B3][B2][B1]------------

2. Non-Interleaved 1F1B (Lower Peak Memory, Medium Bubble)
   GPU 0: [F1][F2][F3][F4][B1][F5][B2][F6][B3]...
   GPU 1: ----[F1][F2][F3][B1][F4][B2][F5][B3]...
   GPU 2: --------[F1][F2][B1][F3][B2][F4][B3]...
   GPU 3: ------------[F1][B1][F2][B2][F3][B3]...

3. Interleaved 1F1B (Megatron-LM Style - Minimal Bubble & Low Memory)
   * Each GPU holds v virtual stages (e.g., v = 2) *
   GPU 0: [F1v1][F1v2][F2v1][B1v1][F2v2][B1v2]...
   GPU 1: ------[F1v1][F1v2][B1v1]...
```

---

## 3. Advanced Strategy: Interleaved 1F1B Scheduling

Megatron-LM ne **Interleaved 1F1B** introduce kiya tha:

1. Har physical GPU ko ek continuous chunk ke bajaye multiple non-contiguous **virtual stages** ($v$) assign kiye jaate hain.
   * E.g., 8-layer model on 2 GPUs ($p=2, v=2$):
     * **GPU 0:** Stage 1 (Layers 1-2) aur Stage 3 (Layers 5-6)
     * **GPU 1:** Stage 2 (Layers 3-4) aur Stage 4 (Layers 7-8)
2. Isse forward/backward steps ke switching overhead badh jaata hai lekin overall pipeline bubble significantly cut ho jaata hai:

$$\text{Interleaved Bubble Fraction} \approx \frac{1}{v} \times \frac{p - 1}{m}$$

| Metric | GPipe | Standard 1F1B | Interleaved 1F1B |
| :--- | :--- | :--- | :--- |
| **Bubble Time** | $\frac{p-1}{m+p-1}$ | $\frac{p-1}{m}$ | $\frac{p-1}{v \cdot m}$ |
| **Activation Memory** | High ($O(m)$) | Medium ($O(p)$) | Low ($O(p)$) |
| **P2P Transfers** | Low | Low | High ($v \times$ transfers) |

---

## 4. Activation Recomputation (Checkpointing) in PP

Bubble reduce karne ke liye $m$ ko badhane par activation memory **Out of Memory (OOM)** error de sakti hai. Isko solve karne ke liye **Activation Checkpointing** use hota hai:

```
Standard Flow:   [Forward Step] ──(Save ALL Intermediate Activations)──► [Backward Step] (High HBM Usage)

Recomputation:   [Forward Step] ──(Save ONLY Boundary Inputs)──────────► [Backward Step]
                                                                                │
                                                   (Re-run Forward on-the-fly) ◄┘ (Low HBM, +33% Compute)
```

---

## 5. Day 19 Summary Checklist

1. **Bubble Metric:** Pipeline idle time depends directly on $\frac{p-1}{m}$.
2. **1F1B Advantage:** Standard 1F1B memory peak ko restrict karta hai $p$ micro-batches tak, whereas GPipe scales with $m$.
3. **Interleaved Factor:** $v$ virtual stages use karke bubble fraction ko $\frac{1}{v}$ times shrink kiya ja sakta hai at the cost of more frequent P2P communication.