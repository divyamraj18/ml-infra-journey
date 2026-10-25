# Assignment 1.1 result — Colab T4

```text
n=1048576 blocks=4096 threads_per_block=256
kernel 0.175 ms, effective 66.84 GiB/s, mismatches 0
```

GPU: Tesla T4, compute capability 7.5, Turing. Memory **16 GB GDDR6**, peak bandwidth **320 GB/s**. T4 pe HBM nahi hoti. HBM A100 / H100 / B200 pe hai.

## Launch

`n = 1 << 20 = 1,048,576`. Block size 256. `1048576 / 256 = 4096` blocks exactly. Last block mein extra threads nahi the, `i < n` ne kisi thread ko nahi kaata.

`mismatches 0` ka matlab har `c[i] = a[i] + b[i]` host ke expected value se match hua. Kernel sahi hai.

## 66.84 GiB/s ka matlab

Kernel 2 arrays padhta hai aur 1 likhta hai.

```text
moved = 3 * 1,048,576 * 4 bytes = 12,582,912 bytes ≈ 12 MiB
time  = 0.175 ms
66.84 GiB/s ≈ 72 GB/s
```

T4 ka peak 320 GB/s hai. Achieved bandwidth peak ka lagbhag **22%** hai. Element-wise add memory-bound hai, lekin yeh run peak ke kareeb nahi pahuncha.

Do wajah:

1. **Kernel bahut chhota hai.** 12 MiB, 0.175 ms. GPU clock ramp aur launch ka fixed cost itne chhote kernel pe bada hissa le leta hai. Bandwidth tab saturate hoti hai jab transfer lambe samay tak chalta rahe.
2. **Load `float` ka hai, `float4` ka nahi.** Har thread 4 bytes maangta hai. Peak ke kareeb jaane ke liye vectorized load (`float4`, 16 bytes per thread) use hota hai. Woh optimization abhi nahi kari.

Colab T4 pe 72 GB/s ek sahi, chhota, coalesced streaming kernel hai. H100 ke ~3.35 TB/s se compare mat karo. Woh alag memory system hai.
