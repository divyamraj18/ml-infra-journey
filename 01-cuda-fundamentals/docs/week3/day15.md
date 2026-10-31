# Day 15: Data parallel, sync SGD, aur do profilers

Socho 4 teachers ek hi syllabus padha rahe hain, lekin har teacher ke paas alag answer sheets hain. Week ke end mein woh apne "kya change karna hai" notes jodte hain.

Yeh **data parallelism** hai. Model ki copy har GPU pe **poori** hoti hai. Batch ke tukde alag hote hain. Har GPU apne tukde pe gradient nikalta hai. Phir gradients judte hain.

**Synchronous SGD.** Sab GPUs apna batch khatam karte hain. Gradients **AllReduce** se sum hote hain (Week 2). Phir sab **same** update lete hain. Step ke baad weights match karte hain. LLM training yahi use karti hai.

**Asynchronous SGD.** Koi GPU doosre ka wait nahi karta. Woh purana gradient bhej deta hai. Update **stale** ho sakta hai: step 10 ka gradient step 12 ke weights pe lag raha hai. Tez dikhta hai. Convergence noisy hoti hai.

**Parameter server.** Ek machine weights rakhti hai. Workers gradients **push** karte hain, naye weights **pull** karte hain. Server ki network bottleneck ban jaati hai. **AllReduce** mein koi server nahi hota. Ring pe har link same kaam karti hai.

Do profilers, do sawal:

| Tool | Sawal | Kya dikhta hai |
|---|---|---|
| **Nsight Systems** `nsys` | GPU idle kyun hai? | Poori timeline: kernels, copies, gaps, NCCL |
| **Nsight Compute** `ncu` | Yeh ek kernel slow kyun hai? | Occupancy, stall reason, memory throughput |

Day 6 ka PyTorch Profiler op ka **naam** batata hai. `nsys` system ki timeline hai. `ncu` ek kernel ka microscope hai.

**Rule:** sync data parallel = full model on every GPU + AllReduce of gradients. `nsys` gaps dhundhta hai. `ncu` ek kernel ka stall dhundhta hai.
