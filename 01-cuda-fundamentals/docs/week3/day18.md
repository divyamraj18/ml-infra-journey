# Day 18: Pipeline parallel aur 1F1B

Data parallel model ki **copy** banata hai. Model ek GPU mein na ghuse (Week 1: 7B Adam ≈ 112 GB), to layers kaat do.

GPU0 layers 0–7 rakhe. GPU1 layers 8–15. Yeh **pipeline parallelism (PP)** hai. Activation ek stage se agle stage ko jaati hai. Poora gradient AllReduce nahi hota. Sirf boundary pe activation jaati hai.

Ek bada batch ek saath mat bhejo. Use **microbatches** mein todo. GPU0 microbatch 1 forward karke GPU1 ko bhejta hai, aur turant microbatch 2 shuru kar deta hai. Warna GPU1 tab tak khali hai jab tak poora batch GPU0 pe na ho.

**1F1B** (one forward, one backward): pipeline bhar jaane ke baad har stage ek forward aur ek backward alternate karta hai.

Agar saare microbatches ka forward pehle khatam karo, backward baad mein, to beech ki activations bahut der tak memory mein rehti hain. 1F1B unhe jaldi free karata hai. Memory kam. Bubble (Day 19) ab bhi hai.

**Rule:** PP layers split karta hai. Microbatches pipeline ko bharte hain. 1F1B activation memory ko control karta hai.
