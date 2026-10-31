# Day 28: Prefill, decode, KV cache, batching

Ek request do phase mein chalti hai.

**Prefill.** Poora prompt ek saath. Har token doosre ko dekh sakta hai. Math bahut hai. Aksar **compute-bound**. User is wait ko **TTFT** (time to first token) kehta hai.

**Decode.** Agla token ek-ek nikalta hai. Har step pe saare weights HBM se aate hain aur unpe chhoti math hoti hai (Day 2). **Memory-bound**. Tokens ke beech ka gap **ITL** (inter-token latency) hai.

**KV cache.** Pichle tokens ki Key aur Value dobara compute mat karo. Unhe HBM mein rakh lo.

```text
bytes ≈ 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_number
```

`2` K aur V dono ke liye hai. Sequence lambi, cache linear badhti hai. Weights fixed hain. Lambi context pe cache weights se badi ho jaati hai. Yeh inference OOM hai. Training wali activation OOM nahi.

**PagedAttention (vLLM).** OS ke pages jaisa. KV cache ko ek lamba reserved array mat banao. Fixed-size **blocks** ki list rakho. Request khatam, blocks pool mein wapas. Beech ki khali jagah (fragmentation) kam.

**Continuous batching.** Purana batch: sab requests ek saath shuru, sab khatam, phir agla batch. Chhoti request lambi request ka wait karti hai. Iteration-level batching: jo request khatam ho, uski slot nayi request le leti hai. GPU khali slot nahi gawaata.

**Speculative decoding.** Chhota draft model 4 tokens guess karta hai. Bada model unhe ek prefill-style step mein check karta hai. Guess sahi ho to 4 tokens ek verify mein. ITL girti hai. Galat guess pe rewind.

**Rule:** prefill compute hai, decode bandwidth hai. KV cache sequence ke saath badhti hai. Pages aur continuous batching HBM ko requests ke beech reuse karate hain.
