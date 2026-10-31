# Day 35: Failure ka bill, aur Triton

1024 nodes, 3 mahine ki training. Ek node mar jaaye to uske gradients gaye. Sync AllReduce atak jaata hai, kyunki ring mein ek rank missing hai.

Last checkpoint 30 minute purana ho to 1024 GPUs 30 minute ka kaam dobara karte hain.

```text
cost ≈ 1024 × 0.5 hour × price of one GPU-hour
```

Checkpoint aur fast restart isi bill ko chhota karte hain.

**Triton Inference Server** sirf HTTP server nahi hai. Woh requests queue karta hai. **Dynamic batching** same model ki requests jodta hai, phir GPU pe ek batch bhejta hai. Batch full ho ya timeout ho, tab kernel chalta hai.

**Ensemble.** Model A ka output model B ko, server ke andar, client ke bina. Preprocess, phir model, phir postprocess.

**Shared memory IPC.** Client aur Triton same machine pe hon to HTTP pe tensor copy mat karo. `/dev/shm` mein buffer rakho, sirf handle bhejo.

**PTQ vs QAT.**

| | Kab | Cost |
|---|---|---|
| **PTQ** | Training ke baad. AWQ, GPTQ, FP8 calibration. | Sasta. Thodi accuracy jaati hai. |
| **QAT** | Training ke dauran fake quant noise. | Mehnga. Accuracy better reh sakti hai. |

AWQ important weights ko kam quantize karta hai. GPTQ error dekh kar weights adjust karta hai. FP8 hardware path Day 5 aur Day 6 pe hai.

**Rule:** failure ka bill `GPUs × time since checkpoint` hai. Triton batch banata hai. PTQ sasta hai. QAT tab jab accuracy na bache.
