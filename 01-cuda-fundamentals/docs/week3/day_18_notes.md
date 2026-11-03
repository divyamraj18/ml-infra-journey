# 📅 Day 18: Pipeline Parallelism (PP) & Interleaved 1F1B Scheduling

Aaj ke day ka primary goal hai **Pipeline Parallelism (PP)** ke mechanics ko samajhna: Jab model size single GPU ki memory capacity se bada ho jata hai, toh model ki sequential layers ko multiple GPUs par kaise split karte hain, activation transfer kaise kaam karta hai, aur **Interleaved 1F1B (One Forward, One Backward)** schedule bubbles ko kaise minimize karta hai.

---

## 1. Pipeline Parallelism (PP) Kya Hai?

Jab koi LLM (jaise 70B+ parameters) single GPU memory (HBM) me fit nahi aata, tab Tensor Parallelism (intra-node) ke alawa hum layers ko sequentially alag-alag GPUs me distribute karte hain.

* **Layer Splitting:** E.g., 48 Transformer layers ko 4 GPUs par split kiya jata hai:
  * **GPU 0:** Layers 1–12
  * **GPU 1:** Layers 13–24
  * **GPU 2:** Layers 25–36
  * **GPU 3:** Layers 37–48

### Sequential Layer Splitting Visualized

```
[ Input Batch X ]
       │
       ▼
┌──────────────┐    Activations (P2P)    ┌──────────────┐
│    GPU 0     │ ───────────────────────►│    GPU 1     │
│ (Layers 1-12)│                         │(Layers 13-24)│
└──────────────┘                         └──────┬───────┘
                                                │ Activations (P2P)
                                                ▼
┌──────────────┐    Activations (P2P)    ┌──────────────┐
│    GPU 3     │ ◄───────────────────────│    GPU 2     │
│(Layers 37-48)│                         │(Layers 25-36)│
└──────┬───────┘                         └──────────────┘
       │
       ▼
   [ Loss ]
```

---

## 2. Naive Pipeline Execution & The Bubble Problem

Agar hum poore global mini-batch ko ek saath pipeline se pass karein, toh downstream GPUs pehle upstream GPUs ka wait karte rahenge (idle time).

### Naive Execution Timeline (The Giant Bubble)

```
GPU 0 | F1 | -- | -- | -- | -- | -- | -- | B1 |
GPU 1 | -- | F1 | -- | -- | -- | -- | B1 | -- |
GPU 2 | -- | -- | F1 | -- | -- | B1 | -- | -- |
GPU 3 | -- | -- | -- | F1 | B1 | -- | -- | -- |
        ◄──────── IDLE BUBBLE ────────►
```

* **Pipeline Bubble:** Downtime jahan GPUs inactive baithte hain data ka wait karte hue.
* **Solution:** Global batch ko $m$ **Micro-batches** me slice kar diya jata hai.

---

## 3. Micro-batching & Schedules: GPipe vs 1F1B

### A. GPipe Schedule (All Forwards -> All Backwards)

GPipe me saare micro-batches ke forward passes pehle run hote hain, fir backward passes execute hote hain.

```
GPU 0 | F1 | F2 | F3 | F4 | -- | -- | -- | B4 | B3 | B2 | B1 |
GPU 1 | -- | F1 | F2 | F3 | F4 | -- | B4 | B3 | B2 | B1 | -- |
GPU 2 | -- | -- | F1 | F2 | F3 | F4 | B4 | B3 | B2 | B1 | -- |
```
* **Problem:** Substantial activation memory stash karna padta hai jab tak backward pass start na ho.

---

### B. 1F1B (One Forward, One Backward) Schedule

1F1B memory peak ko restrict karne ke liye **Warmup Phase** ke baad Har Ek Forward step ke baad immediate Ek Backward step trigger karta hai.

1. **Warmup Phase:** Upstream GPUs initial micro-batches ($F$) execute karte hain to fill the pipeline.
2. **1F1B Steady State:** Alternate 1 Forward step and 1 Backward step for ready micro-batches.
3. **Cool-down Phase:** Drain remaining backward passes.

### 1F1B Execution Timeline

```
GPU 0 | F1 | F2 | F3 | F4 | B1 | F5 | B2 | F6 | B3 | B4 | B5 | B6 |
GPU 1 | -- | F1 | F2 | F3 | B1 | F4 | B2 | F5 | B3 | B4 | B5 | -- |
GPU 2 | -- | -- | F1 | F2 | B1 | F3 | B2 | F4 | B3 | B4 | -- | -- |
GPU 3 | -- | -- | -- | F1 | B1 | F2 | B2 | F3 | B3 | F4 | B4 | -- |
```

---

## 4. Interleaved 1F1B (Megatron-LM Style)

Standard 1F1B me har GPU ko ek continuous chunk of layers (e.g., GPU 0 gets layers 1-8) milta hai.

**Interleaved 1F1B** me har GPU multiple non-contiguous virtual stages hold karta hai:
* **GPU 0:** Layers 1–4 AND Layers 17–20
* **GPU 1:** Layers 5–8 AND Layers 21–24
* ...

```
Virtual Stage Assignment:
GPU 0 ──► Stage 1 (Layers 1-4)  ────────┐
GPU 1 ──► Stage 2 (Layers 5-8)          │ (Activation pass)
GPU 0 ──► Stage 3 (Layers 9-12) ◄───────┘
```

### Advantages:
* Pipeline bubble fraction drastically drop hota hai without increasing micro-batch counts:
$$\text{Bubble Time Fraction} \approx \frac{p - 1}{v \cdot m}$$
  *(where $p$ = pipeline stages, $v$ = virtual stages per GPU, $m$ = micro-batches).*

---

## 5. Summary Checklist for Day 18

1. **P2P Communication:** Pipeline Parallelism uses Point-to-Point (P2P) `send`/`recv` primitives across adjacent ranks instead of collective operations like AllReduce.
2. **Micro-batching:** Splitting global batches reduces idle hardware time.
3. **1F1B Optimization:** Limits peak memory usage by immediately executing backward passes to release intermediate activation buffers.