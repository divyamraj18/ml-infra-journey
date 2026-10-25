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

Socho 8 numbers add karne hain: `1 2 3 4 5 6 7 8`. Har number pe ek insaan kaam karega.

**Thread = ek insaan, ek number.**

Thread 0 sirf `1` dekhta hai. Thread 1 sirf `2` dekhta hai. Kisi ko poori list nahi dekhni. Bas apna ek number.

**Block = ek team.**

8 logon ko ek hi team mat banao. Do teams banao, har team mein 4 log:

- Block 0: jo `1 2 3 4` karte hain
- Block 1: jo `5 6 7 8` karte hain

Ek team ek SM (factory floor) pe jaati hai. Us team ke log ek hi table share kar sakte hain. Doosri team doosre floor pe jaa sakti hai, aur dono teams ek saath kaam kar sakti hain. Doosri team ki table pe yeh team nahi pahunch sakti.

**Grid = saari teams milake, yaani poora kaam.**

Yahan grid = Block 0 + Block 1. Numbers 8 ki jagah 8 million ho jayein, to teams zyada ho jayengi. Idea wahi rehta hai: ek insaan ek number, ek team ek floor, saari teams milake poora job.

## Software level aur hardware level

Thread, block, aur grid **software** hain. Tum decide karte ho. SM, CUDA core, aur warp **hardware** hain. Chip pe pehle se bane hain. Tum unki ginti nahi badal sakte.

12 numbers, har team mein 4 log: tumne **3 blocks** banaye. Yeh software ka faisla hai. GPU pe shayad 100 floors (SMs) hon. Tumne 100 floors nahi banaye. Tumne 3 teams banayi, aur GPU un teams ko khali floors pe bitha deta hai.

| Software (tum likhte ho) | Hardware (chip pe pehle se hai) | Kaise milte hain |
|---|---|---|
| **Thread** — ek number pe ek kaam | **CUDA core** — asal worker jo math karta hai | Ek SM ke CUDA cores us block ke threads chalate hain. Threads zyada hote hain, cores kam. Cores baari baari se threads ka kaam karte hain |
| **Block** — ek team | **SM** — ek floor | GPU ek block ko kisi ek SM pe de deta hai. Block SM nahi hai. Block us floor pe mehmaan hai |
| **Grid** — saari teams | **Poora GPU** — saare SMs | Grid ke blocks alag-alag SMs pe bikhar jaate hain, isliye kaam ek saath hota hai |

Ek aur hardware cheez jo tum nahi banate: **warp**. Block ke threads ko hardware khud 32 ke pack mein baant deta hai. Tum block ka size chunte ho. Warp ka size hamesha 32 hota hai. 4 threads wali team mein bhi hardware ek warp banata hai, baaki 28 seats khali reh jaati hain. Isliye real kaam mein team size 32 ka multiple rakhte hain, jaise 128 ya 256.

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
