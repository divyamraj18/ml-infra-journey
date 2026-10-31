# Day 60: Task aur Actor

**Task** `@ray.remote` function hai. Stateless. Same input, naya call, pichli call ki memory nahi. Map aur filter isi ke hain.

**Actor** `@ray.remote` class hai. Stateful. Ek process mein zinda rehta hai. Agli method call pichla `self` dekhti hai. Model server, counter, replay buffer.

Task fail ho to Ray use dobara chala sakta hai, kyunki input se output phir se banta hai. Yeh lineage Week 12 pe hai. Actor ka `self` apne aap wapas nahi aata, jab tak tum checkpoint na likho.

**Rule:** task retry-friendly hai. Actor state hai. State ka backup tumhari zimmedari hai.
