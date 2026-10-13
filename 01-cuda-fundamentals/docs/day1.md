# Day 1: GPU Architecture

Aaj ka sawaal: GPU actually fast kaise hota hai, aur uske andar kaun si cheez kahan baithi hai.

## CPU aur GPU alag kaam ke liye bane hain

CPU **latency** ke liye bana hai. Usme kuch hi powerful cores hote hain, bada cache hota hai, aur woh ek ke baad ek mushkil kaam jaldi khatam karta hai. Jaise OS, networking, ya koi aisi logic jisme agla step pichle step pe depend karta hai.

GPU **throughput** ke liye bana hai. Usme hazaaron chhote cores hote hain. Yeh ek hi instruction ko bahut saare data pieces pe ek saath chalaata hai. Matrix multiply, image processing, neural net layers — yahi pattern hai.

Socho: CPU ek expert engineer hai jo ek mushkil bug akela fix karta hai. GPU ek factory hai jahan hazaaron workers ek hi step baar baar karte hain.

## GPU ke andar kya hai

Ek GPU kai **Streaming Multiprocessors (SMs)** mein banta hai. SM ko factory ka ek floor samjho. Floor ke andar workers aur unka samaan hota hai. Bada warehouse floor ke bahar hota hai.

| Cheez | Simple matlab | Kahan baitha hai |
|---|---|---|
| **CUDA core** | Normal math karta hai: add, multiply | SM ke andar |
| **Tensor core** | Chhoti matrix multiply ek shot mein karta hai (FP16, BF16, FP8) | SM ke andar. ML ka asli accelerator yahi hai |
| **Register** | Ek thread ki private notebook. Sabse tez, bahut chhoti | SM ke andar, har thread ke paas apni |
| **SRAM (shared memory)** | Ek block ke threads ka shared workbench. Tez, lekin sirf kuch KB | SM ke andar |
| **HBM (global memory / VRAM)** | Bada warehouse: model weights aur activations yahan rehte hain. SRAM se bahut slow | Chip ke bahar |

Data HBM se lana mehnga hai. Isliye baad mein hum yeh dekhenge ki GPU kab compute ki wajah se slow hota hai aur kab memory ki wajah se.

## Software kaise is hardware pe baithta hai

Tumhara kaam teen level mein toota hota hai:

1. **Thread** — ek chhota worker. Ek data element pe kaam karta hai.
2. **Block** — threads ka group. Ek block ek SM pe jaata hai. Us block ke threads SRAM share kar sakte hain. Do alag blocks SRAM share nahi kar sakte, kyunki woh alag floors pe ho sakte hain.
3. **Grid** — saare blocks milake. Poora kaam, jaise 10 million numbers add karna.

## Thread ko apna element kaise milta hai

Har thread ko ek number chahiye: "main list ke kaunse element pe kaam karun?"

Sirf `threadIdx` kaafi nahi hai. `threadIdx` sirf **apne block ke andar** ka number hai. Block 0 ke thread 0 aur block 1 ke thread 0 dono ka `threadIdx` 0 hai, lekin woh list ke alag elements pe kaam karte hain.

Isliye teeno chahiye:

- `threadIdx` — is block ke andar main kaunsa thread hoon
- `blockDim` — is block mein total kitne threads hain
- `blockIdx` — grid mein yeh block kaunsa hai

```text
global_id = blockIdx * blockDim + threadIdx
```

Example: har block mein 4 threads hain.

- Block 0, thread 0 → 0 * 4 + 0 = 0
- Block 0, thread 3 → 0 * 4 + 3 = 3
- Block 1, thread 0 → 1 * 4 + 0 = 4
- Block 2, thread 1 → 2 * 4 + 1 = 9

## Warp aur SIMT

Hardware threads ko ek-ek karke nahi chalaata. Woh unhe **32 ke pack** mein chalaata hai. Us pack ko **warp** kehte hain.

**SIMT** ka matlab hai Single Instruction, Multiple Threads. Warp ke 32 threads ek hi instruction ek hi waqt pe chalate hain. Farq sirf itna hai ki har thread apna data use karta hai, apne registers mein.

**Warp divergence** tab hota hai jab ek warp ke kuch threads `if` wali side jaayein aur kuch `else` wali side. Hardware dono ek saath nahi kar sakta. Pehle ek group chalta hai, doosra rukta hai, phir ulta. Throughput girta hai. Isliye GPU pe branching kam rakhte hain, aur jitna ho sake utna same instruction sab threads pe chalate hain.
