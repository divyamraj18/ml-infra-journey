# Day 6: Ampere, Hopper, Blackwell

Weekend ka pehla hissa architecture hai. Cluster mein jo GPUs milti hain woh aksar in teen families ki hoti hain: **Ampere (A100)**, **Hopper (H100 / H200)**, **Blackwell (B200 / GB200)**. Datasheet numbers full SXM / HGX parts ke hain. MIG slice ya PCIe card chhota hota hai.

Whitepaper se har TFLOP yaad nahi karna. Infra ke liye char cheezein track karo:

1. **HBM capacity aur HBM bandwidth** — memory-bound decode isi pe atakta hai.
2. **Tensor Core formats** — kaun sa dtype hardware pe chalta hai: TF32, BF16, FP16, FP8, FP4.
3. **NVLink bandwidth** — tensor parallel aur expert parallel ka AllReduce isi pe atakta hai.
4. **Naye hardware units** — TMA, Thread Block Cluster, Transformer Engine. Yeh CUDA cores se kaam hataate hain.

## Teen generations, ek nazar mein

| | A100 SXM, Ampere | H100 SXM, Hopper | B200, Blackwell |
|---|---|---|---|
| Compute capability | 8.0 | 9.0 | 10.0 |
| Memory | 80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e |
| HBM bandwidth | ~2.0 TB/s | ~3.35 TB/s | ~8 TB/s |
| L2 cache | 40 MB | 50 MB | bada, dual-die |
| Tensor Core | 3rd gen. TF32, BF16, FP16, INT8, FP64 | 4th gen. FP8 add hota hai | 5th gen. FP8, FP6, FP4 |
| FP8 | nahi | haan, E4M3 aur E5M2 | haan, plus FP4 / FP6 microscaling |
| NVLink | 3rd gen, ~600 GB/s | 4th gen, ~900 GB/s | 5th gen, ~1.8 TB/s |
| CPU tak raasta | PCIe | PCIe, ya Grace pe NVLink-C2C | GB200 pe NVLink-C2C |
| Naya unit | async copy global → shared (`cp.async`) | TMA, Thread Block Cluster, Transformer Engine | 2nd gen Transformer Engine, dual-die NV-HBI |

H200 Hopper hi hai, H100 nahi. Memory badal gayi: **141 GB HBM3e, ~4.8 TB/s**. Compute units wahi Hopper ke hain. Capacity wali OOM (Day 3) pe H200 seedha farq dalta hai. Bandwidth wali decode pe bhi.

## Ampere: A100

**Tensor Core** ek chhoti matrix multiply ek instruction mein karta hai. CUDA core ek add ya multiply karta hai. A100 ki 3rd gen Tensor Core pe training ke common formats **FP16, BF16, aur TF32** hain.

**TF32** TensorFloat-32 hai. Exponent FP32 jaisa 8-bit hai, mantissa FP16 jaisa 10-bit. Range FP32 ke kareeb rehti hai, math Tensor Core pe hoti hai. `torch.backends.cuda.matmul.allow_tf32` isi ko on karta hai. FP32 wala kernel Tensor Core use nahi karta. TF32 wala karta hai.

**2:4 structured sparsity:** har chaar weights mein se do zero hon, to Tensor Core us pattern ko tez chala sakta hai. Datasheet ka "with sparsity" wala number isi assumption pe double hota hai. Dense LLM weights pe woh number nahi milta.

**HBM2e ~2 TB/s.** Decode memory-bound hai (Day 2). Token rate is bandwidth se cap hota hai. A100 pe weights padhne ki ceiling H100 se neeche hai, chahe CUDA cores khali kyun na hon.

**NVLink 3 ~600 GB/s** ek GPU se uske NVLink neighbors tak. PCIe Gen4 x16 ~32 GB/s hai, Gen5 x16 ~64 GB/s. Multi-GPU AllReduce NVLink pe chalta hai, PCIe pe nahi.

Ampere ne **`cp.async`** diya: global memory se shared memory tak copy CUDA cores ke address math ke bina, overlap ke saath. Hopper ka TMA isi idea ko hardware unit bana deta hai.

## Hopper: H100

**4th gen Tensor Core FP8 chalaati hai.** Do encodings:

- **E4M3** — 1 sign, 4 exponent, 3 mantissa. Zyada precision, chhoti range (max 448). Forward activations aur weights.
- **E5M2** — 1 sign, 5 exponent, 2 mantissa. Zyada range, kam precision. Gradients, taaki underflow kam ho.

**Transformer Engine** hardware plus library hai. Har tensor ka **amax** (absolute max) track karti hai, usse **scale factor** banati hai, aur tensor ko FP8 mein fit karti hai. Scale kharab ho to woh tensor **BF16** pe reh jaata hai. Day 5 ka FP8 scaling yahin implement hota hai. Adam ki `m` / `v` phir bhi FP32 mein rehte hain.

**TMA, Tensor Memory Accelerator.** Ek copy engine hai. Multi-dimensional tile global memory se shared memory mein kheenchta hai. Threads khud address nahi nikalte. Completion **`mbarrier`** se hoti hai. CUDA cores us dauran math karte rehte hain. Day 4 ka coalesced access ab bhi zaroori hai, lekin bulk tensor move CUDA cores se hata diya gaya hai.

**Thread Block Cluster.** Block se upar, grid se neeche, ek naya scope. Hopper pe cluster mein **16 thread blocks** tak ho sakte hain, paas ke SMs pe saath schedule hote hain. Unke beech **distributed shared memory** hoti hai: ek block, doosre block ki shared memory padh sakta hai, SM-to-SM network pe. Jo data pehle sirf ek block ke SRAM mein band tha, ab cluster ke andar share ho sakta hai.

**HBM3 ~3.35 TB/s** vs A100 ~2.0 TB/s. Memory-bound decode ka ceiling ~1.6× upar. **NVLink 4 ~900 GB/s.** Tensor parallel ka communication term chhota hota hai.

## Blackwell: B200 aur GB200

B200 **do dies** ka ek GPU hai. Beech mein **NV-HBI** hai, NVIDIA ka die-to-die interconnect, itna tez ki software ko ek GPU dikhta hai. Memory **192 GB HBM3e, ~8 TB/s**. A100 se capacity ~2.4×, bandwidth ~4×. Day 3 wali 7B Adam list (112 GB) ek B200 pe ghus sakti hai. 70B ki FP16 weights ~140 GB hain, weights to ghusenge, Adam state nahi. Capacity badhi hai, 16 bytes/parameter wala hisaab nahi badla.

**5th gen Tensor Core** aur **2nd gen Transformer Engine** FP8 ke saath **FP6 aur FP4** chalate hain. **Microscaling:** 32 values ka block ek shared scale factor rakhta hai (OCP MX formats). Har value 4 bit ki ho sakti hai, scale alag se store hota hai. Inference ka memory traffic phir se girta hai. Training ab bhi FP32 master copy rakhti hai.

**NVLink 5 ~1.8 TB/s** per GPU. **GB200** Grace CPU + Blackwell GPUs hai. CPU aur GPU ke beech **NVLink-C2C** hai, ~900 GB/s class, PCIe se kai guna upar. Day 2 wali DRAM → GPU trip ab har machine pe PCIe nahi hai. Grace-Blackwell pe CPU memory GPU ke zyada kareeb hai.

**NVL72** ek rack hai jismein 72 Blackwell GPUs ek hi NVLink domain mein hain. Bada tensor parallel is rack ke andar NVLink pe rehta hai. Bahar jaate hi InfiniBand / RoCE shuru hota hai. Woh Week 2 ka topic hai.

## PyTorch Profiler: weekend ka practical

Architecture padh ke agla sawaal yeh hota hai: meri job in units ko use kar rahi hai ya nahi. **PyTorch Profiler** op-level timeline deta hai. Yeh single-kernel microscope nahi hai.

| Tool | Kya dikhata hai | Kab |
|---|---|---|
| **PyTorch Profiler** (`torch.profiler`, Kineto) | Kaun sa aten op, kaun sa CUDA kernel, CPU aur GPU ke beech gap, `cudaMemcpy`, memory timeline | Day 6–7 |
| **Nsight Systems** (`nsys`) | Poori process ki timeline: kernels, NVTX, OS | Week 3 |
| **Nsight Compute** (`ncu`) | Ek kernel: occupancy, stall reasons, memory throughput | Week 3, Assignment 1.3 |

Profiler mein yeh `activities` hoti hain: `ProfilerActivity.CPU` aur `ProfilerActivity.CUDA`. Trace Chrome tracing ya Perfetto mein khulta hai. Dekhne layak rows:

- **`aten::linear` / `aten::matmul`** — time yahin hai to compute yahin hai.
- **`Memcpy HtoD` / `DtoH`** — CPU DRAM se HBM ki trip. Day 2. Lambi ho to GPU wait kar raha hai.
- **`ncclAllReduce`** — multi-GPU pe NVLink ya PCIe pe collective.
- Kernel name mein `cutlass`, `ampere_sgemm`, `hmma`, `fp8` — dtype aur Tensor Core path ka signal.

**GPU utilization** ka ek number yeh nahi kehta ki Tensor Cores 100% pe hain. Woh kehta hai GPU pe koi kernel schedule ho raha hai. Kernel 100% CUDA time le sakta hai aur phir bhi **memory-bound** ho sakta hai, kyunki HBM bandwidth khatam ho gayi aur Tensor Cores data ka wait kar rahe hain. Woh farq `ncu` ke memory throughput aur stall reasons se pata chalta hai. Aaj itna pehchaan na kaafi hai: timeline mein gap kahan hai, aur busy kernel kis op ka hai.

Script tab likhenge jab tum bolo. Tab `torch.profiler.profile` se ek chhota matmul trace nikalenge aur `Memcpy` vs kernel time khud dekhenge.
