# 📅 Day 23: Tensor Parallelism in MHA (Multi-Head Attention) & MLP Blocks

Aaj ke day ka core focus yeh samajhna hai ki Megatron-LM architecture poore **Transformer Layer** (Multi-Head Attention + MLP / Feed-Forward Network) me Tensor Parallelism ko kaise plug karta hai, bina extra communication steps ke efficiency maintain karte hue.

---

## 1. Multi-Head Attention (MHA) me Tensor Parallelism

Transformer Attention block me 4 primary projections hote hain:
* **$W_Q, W_K, W_V$ (Query, Key, Value Projections):** Inputs ko attention heads me project karte hain.
* **$W_O$ (Output Projection):** Attention output ko original embedding dimension me blend karta hai.

### A. QKV Projections (Column Parallelism)

MHA naturally parallelizable hota hai kyunki har **Attention Head** independent compute karta hai!

1. Multi-Head Attention ke Total Heads $H$ ko Tensor Parallel GPUs $TP\_SIZE$ me distribute kar diya jata hai.
2. $W_Q, W_K, W_V$ matrices ko **Column Parallel** manner me split kiya jata hai.

$$\text{Heads per GPU} = \frac{\text{Total Heads } (H)}{TP\_SIZE}$$

$$\text{GPU 0 gets Column Chunks: } W_Q^0, W_K^0, W_V^0$$
$$\text{GPU 1 gets Column Chunks: } W_Q^1, W_K^1, W_V^1$$

### B. Output Projection (Row Parallelism)

Attention scores aur Softmax compute hone ke baad, saare local heads ke output concatenate hote hain. Is output ko original dimension me project karne wali matrix **$W_O$** ko **Row Parallel** manner me split kiya jata hai.

---

## 2. Megatron-LM MHA Architecture & Communication Reduction

Megatron-LM ka genius architecture design yeh hai ki **QKV (Column Parallel) + Output Projection (Row Parallel)** ko combine karke poor MHA block me **sirf 1 AllReduce** ki zaroorat padti hai!

### Visual Flow Diagram: MHA Tensor Parallel Block

```text
               Input Tensor [B, S, H]  (Identical on all GPUs)
                         │
                         ├─────────────────────────┐
                         ▼                         ▼
                 [ GPU 0 (Rank 0) ]        [ GPU 1 (Rank 1) ]
                 ┌────────────────┐        ┌────────────────┐
                 │ ColumnParallel │        │ ColumnParallel │
                 │  W_Q, W_K, W_V │        │  W_Q, W_K, W_V │
                 └───────┬────────┘        └───────┬────────┘
                         │                         │
                         ▼                         ▼
                  Attention Heads           Attention Heads
                   (Heads 0..N/2)           (Heads N/2..N)
                         │                         │
                         ▼                         ▼
                 ┌────────────────┐        ┌────────────────┐
                 │  RowParallel   │        │  RowParallel   │
                 │      W_O       │        │      W_O       │
                 └───────┬────────┘        └───────┬────────┘
                         │                         │
                         ▼                         ▼
                   Partial Output            Partial Output
                         │                         │
                         └───────────┬─────────────┘
                                     ▼
                            [ NCCL AllReduce ] (SUM)
                                     │
                                     ▼
                   Final Attention Output [B, S, H]
```

---

## 3. MLP / Feed-Forward Network (FFN) Block me TP

Transformer ka MLP Block do linear layers se mil kar banta hai (SwiGLU / GeLU activations ke saath):
1. **$W_1$ (Up-Projection / Gate):** Hidden dimension ko expand karta hai (e.g., $h \to 4h$).
2. **$W_2$ (Down-Projection):** Hidden dimension ko shrink karke standard size par lata hai (e.g., $4h \to h$).

### Optimal Split Strategy:

* **$W_1$ (Up-Projection):** **Column Parallel** split ($4h / TP\_SIZE$). Activation function (e.g., GELU) local outputs par independently execute hota hai without communication!
* **$W_2$ (Down-Projection):** **Row Parallel** split ($4h / TP\_SIZE \to h$).

### Communication Overhead in MLP Block:
```text
Input [B, S, H] ──► [ ColumnParallel W1 ] ──► [ GeLU ] ──► [ RowParallel W2 ] ──► [ AllReduce ] ──► Output [B, S, H]
```
> **Key Note:** Poore MLP Block me **sirf 1 AllReduce** lagta hai (RowParallel $W_2$ ke output par).

---

## 4. End-to-End Transformer Layer Tensor Parallel Flow

Ek poore Transformer Block (Attention + MLP) me Total Communication overhead:

```text
           Input Activations X
                   │
                   ▼
┌──────────────────────────────────────┐
│     Tensor Parallel MHA Block        │
│  (Column W_qkv ──► Row W_out)        │
└──────────────────┬───────────────────┘
                   │
                   ▼
         [ 1st NCCL AllReduce ]  <── Communication Point 1
                   │
                   ▼
             LayerNorm / Add
                   │
                   ▼
┌──────────────────────────────────────┐
│     Tensor Parallel MLP Block        │
│  (Column W_1 ──► GeLU ──► Row W_2)   │
└──────────────────┬───────────────────┘
                   │
                   ▼
         [ 2nd NCCL AllReduce ]  <── Communication Point 2
                   │
                   ▼
             LayerNorm / Add
```

* **Total AllReduce Calls per Transformer Layer:** Exactly **2 AllReduce** operations per layer (Forward Pass me 2, Backward Pass me 2 AllReduce/AllGather).

---

## 5. Tensor Shape Tracking Table

Assumptions: Batch Size = $B$, Sequence Length = $S$, Hidden Dim = $H$, Total Heads = $N_{heads}$, $TP\_SIZE = 2$.

| Layer / Step | Input Tensor Shape | Weight Matrix Shape per GPU | Output Tensor Shape per GPU | Communication Required |
| :--- | :--- | :--- | :--- | :--- |
| **QKV Projection** | `[B, S, H]` | `[H, (3 * H) / 2]` | `[B, S, 3 * H / 2]` | **None** |
| **Self-Attention** | `[B, S, H/2]` | N/A (Compute Scores) | `[B, S, H/2]` | **None** |
| **Output Projection ($W_O$)** | `[B, S, H/2]` | `[H/2, H]` | `[B, S, H]` *(Partial)* | **AllReduce (Sum)** |
| **MLP Layer 1 ($W_1$)** | `[B, S, H]` | `[H, 4H/2]` | `[B, S, 2H]` | **None** |
| **MLP Layer 2 ($W_2$)** | `[B, S, 2H]` | `[2H, H]` | `[B, S, H]` *(Partial)* | **AllReduce (Sum)** |

---

## 6. Summary Checkpoints for Day 23

1. **Head Partitioning:** QKV matrices ko heads ke dimension par split kiya jata hai, allowing zero communication during attention score computation.
2. **Minimal Communication:** Megatron-LM design ek poore Transformer layer me communication ko **sirf 2 AllReduce operations** tak restrict kar deta hai.
3. **Activation Compatibility:** GeLU / SwiGLU activation functions ColumnParallel layer ke output par local GPU me calculate hote hain, jisse communication overhead bypass hota hai.