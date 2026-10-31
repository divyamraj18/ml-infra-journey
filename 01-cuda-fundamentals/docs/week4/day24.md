# Day 24: ZeRO-1, ZeRO-2, ZeRO-3

Week 1 ka hisaab: Adam mixed precision ≈ **16 bytes per parameter**. 7B × 16 = 112 GB, activations se pehle.

Data parallel yeh 16 bytes **har** GPU pe rakhta hai. 8 GPUs pe 8 copies. Weights same hain. Sirf data alag hai. Copies waste hain.

**ZeRO** (Zero Redundancy Optimizer) copies kaat ta hai. Har GPU sirf `1/N` state rakhta hai.

| Stage | Kya shard hai | Har GPU pe kya full rehta hai |
|---|---|---|
| **ZeRO-1** | Optimizer states: `m`, `v`, FP32 copy | Weights aur gradients |
| **ZeRO-2** | Optimizer + gradients | Sirf weights. Backward ke baad **ReduceScatter**. |
| **ZeRO-3** | Optimizer + gradients + weights | Kuch full nahi. Forward se pehle **AllGather**, use karo, hata do. |

ZeRO-3 aur PyTorch **FSDP** same family hain. Communication: har step parameters ka AllGather, gradients ka ReduceScatter. Data parallel ka ek AllReduce isse kam baar bolta hai. Memory jeet tab hai jab model ek GPU pe nahi ghusta.

**Rule:** ZeRO redundancy kaat ta hai, FLOPs nahi. Stage jitni upar, memory utni kam, collectives utne zyada.
