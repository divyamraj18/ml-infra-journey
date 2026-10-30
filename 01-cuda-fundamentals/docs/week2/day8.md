# Day 8: NVLink, NVSwitch, coalescing, bank conflicts, tiled GEMM

Week 1 mein traffic ek GPU ke andar tha: DRAM, phir PCIe, phir HBM, phir SRAM, phir registers. Is hafte do sawal hain.

1. Ek box ke andar 8 GPUs ek doosre se kaise baat karte hain?
2. Ek GPU memory se data kaise uthaye, taaki jo trip lagi, usme har byte kaam aaye?

## Ek box ke andar: PCIe, NVLink, NVSwitch

**Intra-node** ka matlab ek hi machine, ek hi chassis. DGX H100 uska example hai: 8 H100 GPUs ek box mein.

Socho 8 factories ek campus mein hain. Unke beech samaan bhejne ke teen raaste hain.

**PCIe** campus ki sadharan sadak hai. GPU ko CPU aur NIC se yahi jodti hai. Gen4 x16 lagbhag 32 GB/s per direction hai. Gen5 x16 lagbhag 64 GB/s. Do H100 ki HBM lagbhag 3.35 TB/s hai. Agar woh sirf is sadak se baat karein, to factories ka maal sadak pe atak jaata hai. GPU idle baithta hai.

**NVLink** factories ke beech ki dedicated expressway hai. PCIe se kai guna chaudi.

| GPU | NVLink generation | Bidirectional bandwidth |
|---|---|---|
| A100 | 3rd | ~600 GB/s |
| H100 | 4th | ~900 GB/s |
| B200 | 5th | ~1.8 TB/s |

H100 pe 18 NVLink 4 links hote hain. Ek link lagbhag 25 GB/s per direction hai. `18 × 25 × 2 = 900 GB/s` bidirectional. Yeh Day 6 wala number hai.

**NVSwitch** interchange hai. Bina switch ke expressway sirf padosi factory tak jaati hai. Switch har factory ko har doosri factory se full speed pe jod deta hai.

DGX H100:

- 8 H100 GPUs
- 4 third-generation NVSwitch chips
- GPU seedha GPU se wire nahi hoti. Har GPU apni NVLink links switch mein daalti hai.
- Koi bhi pair, jaise GPU3 aur GPU7, NVSwitch ke through ~900 GB/s pe baat kar sakta hai.

Yeh sirf box ke andar hai. Box ke bahar NVLink khatam. Agle din ka InfiniBand wahan se shuru hota hai.

```text
        NVSwitch  NVSwitch  NVSwitch  NVSwitch
          |  \      |  \      |  \      |  \
        GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
          |    |    |    |    |    |    |    |
         PCIe / NIC   ... yeh links box se bahar jaati hain. Yeh NVLink nahi hai.
```

## Coalescing: warehouse aur truck

GPU ki global memory (HBM) ek bada warehouse hai. 32 threads ek saath samaan lene aate hain. Un 32 threads ko **warp** kehte hain.

Warehouse se samaan lane ke liye truck ka size fixed hai. Us truck ko **sector** kehte hain. Size **128 bytes** hai.

Truck chahe 1 float (4 bytes) uthaye ya 32 floats (128 bytes), trip ka cost same hai. Ek **memory transaction**. Isliye hum chahte hain ki ek trip mein jitna samaan aaye, sab use ho jaaye. Isi ko **coalesced access** kehte hain. Truck aadha khali gaya to **bandwidth waste** hoti hai.

### Rule 1: merge hardware karta hai

32 threads khud aapas mein merge nahi karte. Woh ek saath request bhejte hain. GPU ka hardware dekhta hai ki unka data kin 128-byte sectors mein padta hai, aur un requests ko jod deta hai.

### Rule 2: alignment, 128-byte boundary

Memory 128-byte ke blocks mein kati hui hai: 0–127, 128–255, 256–383, aage bhi.

- Pehla thread aisi address se maange jo 128 se divide ho jaati hai (aligned). Saara data ek sector mein fit hota hai. **1 transaction.**
- Pehla thread beech se shuru kare, maan lo address 64. Tab 128 bytes do sectors mein fail jaate hain: 64–127 pehle sector, 128–191 agla sector. Isko **misaligned access** kehte hain. Hardware ko **2 transactions** karni padti hain. Time waste hota hai.

### Rule 3: stride

**Stride** do consecutive threads ke beech kitne elements ka gap hai. Warp mein threads `t = 0` se `31` hain. Thread `t` element `t * stride` padhta hai.

**Stride = 1. Best case. Coalesced.**

Thread 0 element 0 padhta hai, thread 1 element 1, thread 31 element 31. Sab consecutive hain. 32 floats × 4 bytes = 128 bytes. Ek hi sector.

**Result:** 1 memory transaction. Har byte useful. 0% waste.

**Stride = 2. Aadha waste.**

Thread 0 element 0, thread 1 element 2, thread 2 element 4, thread 31 element 62. Beech ke floats chhoot gaye. Jo 128 bytes truck laaya, usme se aadhe threads ne maange hi nahi.

**Result:** 50% bandwidth waste.

**Stride = 32. Worst case. Uncoalesced.**

Thread 0 element 0, thread 1 element 32, thread 2 element 64, thread 31 element `31 × 32 = 992`. Har thread alag sector pe hai.

**Result:** 32 alag transactions. Bus `32 × 128` bytes move karti hai. Kernel sirf `32 × 4` bytes use karta hai. Sabse slow case.

### Rule 4: row-major matrix

2D matrix memory mein 1D line ki tarah store hoti hai. Pehle row 0 ke saare elements, phir row 1, phir row 2. Isko **row-major** kehte hain.

**Row ke along. Fast.**

Ek hi row ke consecutive columns: `(row 0, col 0)`, `(row 0, col 1)`, `(row 0, col 2)`. Memory mein yeh chipke hue hain. **Stride = 1.** Fully coalesced.

**Column ke along. Slow.**

Ek hi column: `(row 0, col 0)`, `(row 1, col 0)`, `(row 2, col 0)`. Do rows ke beech poori row ki width ka gap hai. Row width 4096 ho to thread 0 aur thread 1 ke beech 4096 elements hain. **Stride = 4096.** Har thread alag sector.

**Result:** fully uncoalesced. Kernel sahi jawab dega, speed gir jaayegi.

Assignment 1.1 isi rule pe tha. Thread `i` ne `a[i]` padha, stride 1. T4 pe 66.84 GiB/s coalescing tootne se nahi aaya. Woh run peak se door isliye tha kyunki buffer chhota tha (12 MiB) aur har load ek `float` tha, `float4` nahi.

## Shared memory: 32 cash counters

Global memory ka truck HBM se aata hai. **Shared memory** SM ke andar ki SRAM hai, bahut tez, bahut chhoti. Yahan bhi ek warp ek saath padhta hai, lekin rule alag hai. Yahan **bank conflict** hota hai.

Socho 32 cash counters hain. Inhe **banks** kehte hain. 32 threads ek saath paise nikalne aaye hain.

- Ek counter pe ek time pe ek insaan deal karta hai, jab addresses alag hon.
- Har bank 4 bytes ka hai, ek 32-bit word.
- Bank nikalne ka formula:

```text
word = byte_offset / 4
bank = word % 32
```

### Chaar cases

**1. Conflict-free. Best case.**

32 threads 32 alag banks pe jaate hain. Thread `t` reads `smem[t]`.

**Result:** ek cycle, ek transaction.

**2. Broadcast. Conflict nahi.**

Saare 32 threads ek hi address padhte hain, `smem[0]`. Hardware ek value uthakar sabko de deta hai.

**Result:** 1 cycle. Yeh conflict nahi hai.

**3. Bank conflict. Worst case.**

Threads alag data maang rahe hain, lekin woh data ek hi bank pe pada hai. Thread `t` reads `smem[t * 32]`. Saare addresses bank 0 pe jaate hain, words alag hain.

**Result:** **32-way conflict.** Hardware ek-ek karke 32 serial transactions karta hai. Code **galat nahi** hota. Speed girti hai. GPU ki SRAM bandwidth waste hoti hai.

**4. Multicast.**

Kuch threads same address pe hain, baaki alag banks pe. Sirf distinct banks count hote hain. Jitne unique banks, utni cycles.

### Padding: extra column

Matrix multiply ya transpose mein yeh problem tab aati hai jab tile **32 columns** wide ho aur tum column ke along neeche padho.

`index = row * 32 + col` mein agli row ka wahi column `32` words door hai. `32 % 32 = 0`, isliye har row ka wahi column **same bank** pe girta hai. 32-way conflict.

**Fix:** row ko 32 ki jagah 33 wide banao.

```text
index = row * (32 + 1) + col
```

Agli row ka wahi column 33 words door hai. `33 % 32 = 1`, to woh **agle bank** pe chala jaata hai. Conflicts khatam.

**Trade-off:** ek extra column memory mein waste hota hai. Badle mein serial transactions hat jaate hain aur speed badh jaati hai.

### Kitni shared memory milti hai

Shared memory chip ke andar hai, isliye limited hai.

| GPU | Max shared memory per SM |
|---|---|
| A100 | 164 KB |
| H100 | 228 KB |

Tile is limit mein fit hona chahiye. Bahut bada tile ho to kernel launch fail ho sakti hai, ya ek SM pe kam blocks chalenge. Us girawat ko **occupancy drop** kehte hain.

## Tiled matrix multiplication

`C[M, N] = A[M, K] × B[K, N]`.

`C[i, j]` ek lamba dot product hai: `A` ki row `i` aur `B` ka column `j`.

**Naive.** Ek thread ek `C[i, j]` banata hai aur `k` loop global memory se padhta hai. Warp consecutive columns le to `B` ka load coalesced ho sakta hai, aur `A` ka wahi element broadcast ho sakta hai. Coalescing theek ho sakti hai. Problem **reuse** ki hai.

`A` ka ek element N output columns mein use hota hai. 4096×4096 pe N = 4096. `A` khud `4096 × 4096 × 4 = 64 MB` hai. A100 ki L2 40 MB hai. H100 ki L2 50 MB. Poori `A` L2 mein nahi rehti, isliye wahi element baar baar HBM ke warehouse se aata hai. Paper pe arithmetic intensity high hai. Machine pe kernel memory-bound ho jaata hai.

**Tiling** ka matlab: poora warehouse mat ghumo. Ek chhoti tray workbench (shared memory) pe rakh lo, aur table ke saare threads usi tray ko baar baar use karein.

Block `C` ka ek `BM × BN` tile compute karta hai. K ko `BK` ke steps mein toda jaata hai.

```text
acc = 0
for k0 in 0, BK, 2*BK, ... K:
    coalesced load A[i0 : i0+BM, k0 : k0+BK]  -> shared As
    coalesced load B[k0 : k0+BK, j0 : j0+BN]  -> shared Bs
    __syncthreads()          # sabka load complete
    for kk in 0 .. BK-1:
        acc += As[local_row][kk] * Bs[kk][local_col]
    __syncthreads()          # agli tray se pehle koi purani As na padhe
C[i, j] = acc
```

`__syncthreads()` block ke saare threads ko rokta hai. Yeh warp ka barrier nahi hai. Do blocks SRAM share nahi karte, isliye sync bhi sirf apne block ke andar hai.

Chhota picture. `BM = BN = BK = 2`. Block `C` ke top-left 2×2 pe hai.

```text
k0 = 0:  As = A[0:2, 0:2]    Bs = B[0:2, 0:2]
k0 = 2:  As = A[0:2, 2:4]    Bs = B[2:4, 0:2]
```

`A[0][0]` shared memory mein ek baar aata hai. Us tile ke dono output columns use karte hain. Agla K-step aane tak global se dobara nahi aata. Naive loop yeh reuse L2 ke bharose chhod deta tha.

Global load ab bhi row-wise hai, isliye coalesced reh sakta hai. Column-wise reuse shared memory ke andar hota hai. `BK = 32` bina padding ke column read 32-way conflict hai. Fix wahi `+1` padding hai.

Do tiles ka size, float mein: `2 × BM × BK × 4` bytes. `32 × 32` pe `2 × 32 × 32 × 4 = 8 KB`. 228 KB mein aaram se fit hai.

Assignment 1.2 yahi compare karega: naive vs shared-memory tiled, `4096 × 4096`. Code us assignment mein hoga. Aaj algorithm kaafi hai.

## Aaj ka rule

Box ke andar GPUs **NVLink + NVSwitch** pe bolte hain, PCIe pe nahi. Global memory se warp **128-byte aligned, stride-1** load kare, taaki truck khali na jaaye. Shared memory ke **32 banks** hain. Tile 32 wide ho to ek column pad karo. GEMM ka tile SRAM mein isliye aata hai taaki 64 MB ki matrix L2 se bahar hone ke baad bhi wahi byte HBM se baar baar na aaye.
