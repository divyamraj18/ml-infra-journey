# Day 9: InfiniBand, RoCE, aur TCP kyun nahi chalega

Day 8 box ke andar khatam hota hai. NVSwitch ka aakhri port chassis pe rukta hai. Do DGX machines ke beech ka traffic **inter-node** hai. Uske liye NIC chahiye, aur ek alag network, jise **fabric** kehte hain.

## TCP wala raasta: har parcel post office se

GPU pe gradient HBM mein pada hai. TCP use socket se bhejta hai, aur socket CPU ki DRAM mein rehta hai. Bina RDMA ke parcel ka raasta yeh hai:

```text
GPU0 HBM
  -> cudaMemcpy, PCIe
CPU0 DRAM
  -> kernel TCP stack: copy, checksum, context switch
NIC0
  -> wire
NIC1
  -> kernel TCP
CPU1 DRAM
  -> cudaMemcpy, PCIe
GPU1 HBM
```

Socho parcel seedha factory se factory nahi jaata. Pehle post office (CPU kernel) jaata hai. Wahan copy hoti hai, checksum hota hai, aur har chhoti packet pe syscall hoti hai. Phir doosri factory ke post office, phir wahan ki factory.

Payload **do baar PCIe** pe chalti hai. Latency tens se hundreds of microseconds. Bandwidth NIC ke wire rate pe nahi pahunchti. CPU aur PCIe pe atak jaati hai.

## Training mein yeh har step pe hota hai

Data parallel har step **AllReduce** karta hai. Har GPU apne gradients ka hissa rakhti hai. Unhe sum karke wahi sum har GPU ko wapas dena hota hai. Detail Day 11 pe hai.

7B model, BF16 gradients:

```text
7e9 parameters × 2 bytes = 14 GB
```

Yeh 14 GB har step pe fabric pe move hote hain.

Ring AllReduce, jab GPUs bahut hon, lagbhag `2 × S` bytes bhejta hai. Ek NIC, 400 Gb/s:

```text
S = 14 GB = 112 Gb
2S / 400 Gb/s ≈ 0.56 s
```

Yeh **ek rail, koi overlap nahi**, ek upper-bound sketch hai. Measured step time nahi hai. Asli NCCL isse better karta hai, kyunki node pe 8 NICs hote hain aur collective compute ke saath overlap hota hai.

Point yeh hai: TCP is number ko aur bigaad deta hai, kyunki TCP 400 Gb/s tak pahunchta hi nahi. Compute step 100 ms ka ho aur network 500 ms le, to GPU un 500 ms mein khali baitha rehta hai. Day 2 wala stall, ab HBM ki jagah network.

## InfiniBand: private highway, tokens ke saath

**InfiniBand (IB)** HPC ka apna fabric hai. Ethernet nahi hai.

Link **credit-based** hai. Sender tabhi packet daalta hai jab receiver ne credits (tokens) de diye hon. Switch ka buffer full hone se pehle traffic rukta hai. Packet drop design ka hissa nahi hai. TCP ki tarah drop karke retransmit nahi karna padta.

RDMA ispe native hai. NIC seedha remote machine ki registered memory mein likhta hai. CPU us payload ko haath nahi lagata. RDMA agle din hai.

Jo generations clusters pe dikhengi:

| Generation | Per-port rate | NIC jo aksar saath aata hai |
|---|---|---|
| HDR | 200 Gb/s | ConnectX-6 |
| NDR | 400 Gb/s | ConnectX-7 |
| XDR | 800 Gb/s | next generation |

DGX H100 pe typically **8× ConnectX-7** hote hain, har ek 400 Gb/s NDR, aur **Quantum-2** InfiniBand switches. 8 NICs isliye hain kyunki 8 GPUs hain. Ek GPU ka ek NIC. Us pair ko **rail** kehte hain. Rail Day 12 pe topology banegi.

Chhote messages pe latency ~1 microsecond ki class mein hai. TCP pe woh tens of microseconds hoti hai. Bade gradients pe limit port ki bandwidth hai, syscall nahi.

## RoCE: wahi RDMA, lekin public road pe

**RoCE** = RDMA over Converged Ethernet. Verbs wahi hain: queue pair, write, read. Wire Ethernet hai.

**RoCE v2** UDP/IP mein lipta hota hai, isliye routers cross kar sakta hai. UDP port **4791** hai. RoCE v1 sirf ek L2 subnet mein rehta tha. Clusters v2 use karte hain.

Ethernet default mein packet drop kar sakta hai. RDMA drop pasand nahi karta, kyunki uske paas TCP jaisa reliable byte stream nahi hai. Isliye RoCE ko managed fabric chahiye:

- **PFC** (Priority Flow Control): ek priority ko pause frame se roko, poori link ko nahi. Loss kam hota hai. Galat config pe **PFC storm** aur head-of-line blocking hota hai. Poora cluster pause pe atak sakta hai.
- **ECN**: switch packet drop karne se pehle use mark karta hai, "tum tez ho, dheere chalo."
- **DCQCN**: sender ECN dekh kar apni rate ghatata hai. Yeh RoCE ka congestion control hai.

InfiniBand yeh control credits se khud karta hai. RoCE yeh switch ki config se karta hai. Config sahi ho to bandwidth IB ke kareeb hoti hai. Config kharab ho to training ka loss sahi rehta hai aur step time 2× ho jaata hai. Woh bug Day 12 pe pakda jaata hai.

## Kaun sa fabric

| | InfiniBand | RoCE v2 |
|---|---|---|
| Wire | Apna IB fabric | Ethernet |
| Loss | Credits, drop nahi | PFC + ECN + DCQCN se bachao |
| RDMA | Native | Same verbs, UDP pe |
| Ops cost | Alag switches, alag tooling | Ethernet jo pehle se ho, lekin PFC nazuk hai |
| Kahan dikhega | DGX SuperPOD, NVIDIA reference designs | Bahut se cloud GPU clusters |

Dono ka data path RDMA hai. TCP unmein se koi nahi hai. NCCL log mein `Socket` dikhe to tum in dono pe nahi ho. Woh Day 12 ka pehla check hai.

## Aaj ka rule

NVSwitch box ke andar rehta hai. Box ke bahar **InfiniBand ya RoCE** hai, aur dono RDMA hain. TCP gradient ko CPU DRAM aur kernel se guzaarta hai, isliye step ke dauran GPU network ka intezaar karta hai.
