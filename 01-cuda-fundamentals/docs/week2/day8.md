# Day 8: NVLink, NVSwitch, coalescing rules, bank conflicts, tiled GEMM

Week 1 mein ek GPU ke andar ka traffic tha: DRAM → PCIe → HBM → SRAM → registers. Is hafte do naye sawaal hain. Ek box ke andar 8 GPUs aapas mein kaise bolte hain, aur ek GPU ke andar global memory aur shared memory ko kaise padha jaaye taaki woh bandwidth waste na ho.

## Intra-node: PCIe, NVLink, NVSwitch

**Intra-node** ka matlab ek hi machine, ek hi chassis. DGX H100 uska example hai: 8× H100 ek box mein.

**PCIe** GPU ko CPU aur NIC se jodta hai. Gen4 x16 lagbhag 32 GB/s per direction, Gen5 x16 lagbhag 64 GB/s. Yeh GPU-to-GPU ke liye bahut patla hai. Do H100 agar sirf PCIe se baat karein, to unki 3.35 TB/s wali HBM is link pe baith kar so jaati hai.

**NVLink** seedha GPU-to-GPU link hai.

| GPU | NVLink gen | Bidirectional bandwidth |
|---|---|---|
| A100 | 3rd | ~600 GB/s |
| H100 | 4th | ~900 GB/s |
| B200 | 5th | ~1.8 TB/s |

H100 pe 18 NVLink 4 links hote hain. Ek link lagbhag 25 GB/s per direction hai. `18 × 25 × 2 = 900 GB/s` bidirectional. Yeh Day 6 wala number hai.

**NVSwitch** ek crossbar chip hai. Bina switch ke NVLink sirf kuch padosi GPUs tak jaata hai. Switch har GPU ko har doosri GPU se full bandwidth pe jodta hai.

DGX H100:

- 8× H100
- 4× third-generation NVSwitch
- GPU seedha GPU se wired nahi hota. Har GPU apne NVLink links switch pe daalta hai.
- Koi bhi pair, GPU3 ↔ GPU7, NVSwitch ke through ~900 GB/s pe baat kar sakta hai.

Yeh sirf box ke andar hai. Box ke bahar NVLink khatam. Agle din ka InfiniBand wahan se shuru hota hai.

```text
        NVSwitch NVSwitch NVSwitch NVSwitch
          |  \    |  \     |  \     |  \
        GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
          |    |    |    |    |    |    |    |
         PCIe NIC  ... yeh links box se bahar jaate hain, NVLink nahi
```

## Coalescing ke rules

Day 4: ek warp, 32 threads, ek **128-byte sector**. 32 × 4-byte floats = 128 bytes. Best case ek transaction, 100% bytes useful.

Aaj ke rules:

1. **Warp ek saath issue karta hai.** Hardware 32 alag loads ko merge karke kam se kam 128-byte sectors mein badalta hai. Threads khud merge nahi karte.
2. **Alignment.** Sector 128-byte boundary pe aligned hona chahiye. Agar warp ka pehla address 128 se align nahi hai, to wahi 128 bytes do sectors mein kat jaate hain. Do transactions, kuch bytes waste.
3. **Stride.** Thread `t` element `t * stride` padhe.
   - `stride = 1`: consecutive floats. 1 sector. Coalesced.
   - `stride = 2`: har doosra float. 128 bytes mein se aadhe useful. 50% bandwidth waste.
   - `stride = 32`: har thread alag sector. 32 transactions. 32 × 128 bytes bus pe, kaam ke sirf 32 × 4 bytes.
4. **Row-major matrix.** Memory mein row ke baad row aati hai. Warp agar ek row ke consecutive columns padhe, stride 1 hai. Warp agar ek column padhe, stride = row width hai. 4096-wide row pe stride 4096. Woh uncoalesced hai.

Assignment 1.1 isi rule pe tez tha: thread `i` ne `a[i]` padha, stride 1. 66.84 GiB/s T4 pe isliye aaya, coalescing tootne ki wajah se nahi. Woh run peak se door isliye tha kyunki buffer chhota tha aur load `float` tha, `float4` nahi.

## Shared memory bank conflicts

Shared memory SRAM hai, HBM nahi. Phir bhi warp use ek cycle mein tabhi padhta hai jab 32 threads **32 alag banks** pe jaayein.

32 banks hain. Har bank 4 bytes (ek 32-bit word) wide hai.

```text
word = byte_offset / 4
bank = word % 32
```

| Access | Result |
|---|---|
| Thread `t` reads `smem[t]` | 32 alag banks. Ek transaction. |
| Saare threads same address `smem[0]` | **Broadcast.** Conflict nahi. Ek value sabko milti hai. |
| Thread `t` reads `smem[t * 32]` | Har address bank 0 pe, alag words. **32-way conflict.** 32 serial transactions. |
| Kuch threads same address, baaki alag banks | Multicast. Sirf alag banks matter karte hain. |

Conflict ka matlab galat jawab nahi. Matlab SRAM ka bandwidth serial ho gaya. Kernel sahi rahega, slow hoga.

**Padding** classic fix hai. Tile 32 columns wide ho aur column ke along padhna ho, to `smem[row][32]` mein `row * 32 + col` hamesha same bank pe girta hai kyunki `32 % 32 = 0`. Store karo `smem[row][33]` ke hisaab se:

```text
index = row * (32 + 1) + col
```

Agli row ka same column 33 words door hai. `33 % 32 = 1`, to agli row agle bank pe jaati hai. Ek extra column waste hota hai, conflicts khatam.

H100 pe shared memory **228 KB per SM** tak configurable hai. A100 pe **164 KB**. Tile ko isme fit hona hai, warna launch fail hoti hai ya occupancy girti hai.

## Tiled matrix multiplication

`C[M, N] = A[M, K] × B[K, N]`.

`C[i, j]` poori K-length dot product hai: `A[i, :]` aur `B[:, j]`.

**Naive.** Ek thread ek `C[i, j]` kare, aur `k` loop global memory se padhe. Warp agar consecutive `j` le, to `B[k][j]` coalesced hai aur `A[i][k]` broadcast hai. Coalescing theek ho sakti hai. Problem reuse ki hai.

`A` ka ek element output ke N columns mein use hota hai. 4096×4096 pe N = 4096. `A` khud `4096 × 4096 × 4 = 64 MB` hai. A100 ki L2 40 MB hai, H100 ki 50 MB. Poori `A` L2 mein nahi rehti, isliye wahi element baar baar HBM se aata hai. Arithmetic intensity paper pe high hai, machine pe memory-bound ho jaati hai.

**Tiling** reuse ko SRAM pe force karta hai. Block `C` ka ek `BM × BN` tile compute karta hai. K ko `BK` ke steps mein toda jaata hai.

```text
acc = 0
for k0 in 0, BK, 2*BK, ... K:
    coalesced load A[i0 : i0+BM, k0 : k0+BK]  -> shared As
    coalesced load B[k0 : k0+BK, j0 : j0+BN]  -> shared Bs
    __syncthreads()          # sab threads ka load complete
    for kk in 0 .. BK-1:
        acc += As[local_row][kk] * Bs[kk][local_col]
    __syncthreads()          # agli tile se pehle koi purani As na padhe
C[i, j] = acc
```

`__syncthreads()` block ke saare threads ko rokta hai. Warp-level barrier nahi hai. Do blocks SRAM share nahi karte, isliye sync bhi sirf apne block ke andar hai.

Chhota picture, `BM = BN = BK = 2`. Block `C` ke top-left 2×2 pe hai.

```text
k0 = 0:  As = A[0:2, 0:2]    Bs = B[0:2, 0:2]
k0 = 2:  As = A[0:2, 2:4]    Bs = B[2:4, 0:2]
```

`A[0][0]` shared memory mein ek baar aata hai aur us tile ke dono output columns use karte hain. Global se woh dobara nahi aata, jab tak agla K-step na ho. Yahi reuse naive loop mein L2 ke bharose chhod diya gaya tha.

Global load ab bhi row-wise hai, isliye coalesced reh sakta hai. Column-wise reuse shared memory ke andar hota hai, jahan bank conflicts alag se dekhne hain. `BK = 32` bina padding ke column read 32-way conflict hai. Padding wahi `+1` hai.

Do tiles ka size: `2 × BM × BK × 4` bytes, float mein. `32 × 32` pe `2 × 32 × 32 × 4 = 8 KB`. 228 KB ke andar aaram se. Assignment 1.2 yahi compare karega: naive vs shared-memory tiled, `4096 × 4096`. Code us assignment mein likhenge, aaj algorithm hai.

## Aaj ka rule

Box ke andar GPUs **NVLink + NVSwitch** pe bolte hain, PCIe pe nahi. Ek GPU ke andar warp **128-byte aligned, stride-1** global load kare. Shared memory mein **32 banks** hain. Tile `32` wide ho to pad karo. GEMM ka tile SRAM mein isliye aata hai taaki 64 MB ki matrix L2 se bahar hone ke baad bhi reuse HBM pe na gire.
