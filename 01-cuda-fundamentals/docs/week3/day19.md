# Day 19: Pipeline bubble

**Bubble** woh time hai jab kisi stage ke paas kaam nahi hai. Pichla stage abhi activation bhej raha hai, ya agla stage backward bhej raha hai. GPU on hai, useful math nahi ho rahi.

Rough fraction:

```text
bubble ≈ (number_of_stages - 1) / number_of_microbatches
```

`p` stages, `m` microbatches.

- 8 stages, 8 microbatches: `7/8`. Pipeline adha se zyada wait hai.
- 8 stages, 64 microbatches: `7/64`. Bubble chhota.

Microbatches badhao, bubble ghatta hai. Limit: microbatch bahut chhota ho to GPU pe launch overhead math se bada ho jaata hai.

**Interleaved pipeline.** Ek GPU sirf ek consecutive chunk na rakhe. GPU0 layer 0 aur layer 4 rakhe. GPU1 layer 1 aur layer 5. Logical stages zyada ho jaate hain, bubble chhota. Cost: har GPU pe zyada activation traffic, kyunki ek GPU pipeline mein do jagah baithta hai.

**Rule:** bubble `(p - 1) / m` hai. `m` badhao, ya stages interleave karo. Dono ke paas overhead ya extra traffic ka cost hai.
