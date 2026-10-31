# Day 16: DDP, purane DP se better kyun

Purana `nn.DataParallel` **ek process** hai. GPU0 sabka output gather karti hai. Backward wahi pe heavy hota hai. Baaki GPUs wait karte hain. Python ka GIL bhi usi ek process pe hai.

Socho ek counter hai jahan har bill jama hota hai. Chaar cashiers kaam karte hain, lekin hisaab sirf counter 0 pe banta hai. Counter 0 ki line hi speed hai.

**DDP** (`DistributedDataParallel`) ek process per GPU hai. Har process ke paas poora model hai. Backward ke dauran gradients **buckets** mein AllReduce hote hain, agle layer ke backward ke saath overlap. Koi ek GPU gather point nahi hai.

Backend:

| Machine | Backend |
|---|---|
| GPU | **NCCL** |
| CPU-only local test | **gloo** |

**Rule:** DDP isliye scale karta hai kyunki gradient sync backward ke saath overlap hota hai, aur koi ek GPU boss nahi banta.
