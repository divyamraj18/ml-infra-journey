# Day 17: Local DDP script ka shape

Laptop pe CUDA na ho to bhi shape yahi hai. `torchrun` processes launch karta hai.

```text
torchrun --nproc_per_node=2 train.py
```

Har process ko teen numbers milte hain: `RANK`, `LOCAL_RANK`, `WORLD_SIZE`.

1. `init_process_group(backend="gloo")` CPU pe. GPU pe `"nccl"`.
2. Model ko `DistributedDataParallel` se wrap karo.
3. `DistributedSampler` har rank ko **alag** samples de. Bina sampler ke sab processes same batch padhengi. AllReduce tab same gradient do baar jod dega.
4. `loss.backward()` ke baad DDP khud bucket AllReduce kar leta hai. Seekhne ke alawa manually allreduce mat likho.
5. Sirf **rank 0** checkpoint likhe. Do processes ek file pe likhengi to file corrupt hogi.

Colab T4 pe `nproc_per_node=1` sirf launch test hai. Do GPUs ke bina AllReduce ka time nahi dikhega.

**Rule:** sampler data baante. DDP gradients baante. Rank 0 file likhe.
