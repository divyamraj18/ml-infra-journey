# Day 2: GPU Memory Hierarchy aur Bandwidth

Day 1 mein yeh samajh aa gaya ki kaam threads, blocks, aur grid mein toota hai, aur GPU use SM pe chalaata hai. Aaj ka sawaal: data us waqt kahan pada hai, aur GPU ko us data tak pahunchne mein kitna time lagta hai.

GPU aksar isliye slow nai hota hai kyunki math slow hai. Woh isliye slow hota hai kyunki cores data ka intezaar kar rahe hote hain.

## Teen jagah data reh sakta hai

Door se paas, dheere se tez:

| Jagah | Kahan hai | Kitna bada | Kitna tez | Kya rakha hai |
|---|---|---|---|---|
| **DRAM** | CPU wali machine ki RAM | Bahut bada, sau-sau GB | GPU ke liye sabse slow. Data yahan se PCIe pe chadh kar GPU tak aata hai | Files, batches, jo cheez abhi CPU ke paas hai |
| **HBM** | GPU chip ke bahar chipki hui memory. Isi ko VRAM ya global memory kehte hain | Bada, jaise H100 pe lagbhag 80 GB | DRAM se bahut tez, SRAM se bahut slow. H100 pe lagbhag 3 TB/s | Model weights, activations, KV cache |
| **SRAM** | SM ke andar. Shared memory isi ka hissa hai | Chhoti, ek SM pe sirf kuch sau KB | HBM se kai guna tez | Woh chhota tukda jo abhi baar baar use ho raha hai |
| **Register** | Har thread ki private notebook, SM ke andar | Bahut chhoti | Sabse tez | Ek thread ke 2-3 numbers jo abhi calculation mein hain |

HBM ka pura naam High Bandwidth Memory hai. Naam mein bandwidth hai, latency nahi. Matlab ek second mein bahut saara data nikal sakta hai, lekin pehla byte aane mein phir bhi der hoti hai.

Numbers yaad rakhne hain, order. Exact chip alag ho sakti hai:

- Register: lagbhag 1 clock
- SRAM: kuch clocks
- HBM: sau-sau clocks
- CPU DRAM se GPU tak, PCIe pe: HBM se bhi bahut slow. PCIe Gen4 x16 lagbhag 32 GB/s hai. H100 ki HBM lagbhag 3 TB/s hai. Farq lagbhag 100 guna ka hai

## Bandwidth aur latency alag cheezein hain

**Latency:** pehla byte aane mein kitna time. Ek sawaal poocha, jawab kab aaya.

**Bandwidth:** ek second mein kitne bytes aa sakte hain. Pipeline ek baar bhar jaaye, to uske baad kitna flow hai.

HBM ki latency buri hai, bandwidth achhi hai. GPU isliye hazaaron threads rakhta hai. Jab ek warp HBM ka intezaar kar raha ho, SM doosra warp chala deta hai. Intezar chhupta hai, lekin tabhi, jab paas mein doosra kaam maujood ho. Agar saare warps ek saath data maang rahe hon, to chhupane ke liye kuch bacha nahi. Cores khaali baithe rehte hain.

## Bottleneck: data kitni baar use hota hai

Ek byte HBM se lana mehnga hai. Us byte pe kitni math hoti hai, usse decide hota hai ki GPU busy hai ya intezaar kar raha hai.

**Kam math, zyada aana-jaana.** Do arrays ko element-wise add karna: har output ke liye do numbers padho, ek likho, aur sirf ek add karo. HBM se data laane ka time add karne ke time se bahut zyada hai. CUDA cores baithe rehte hain. Is ko **memory-bound** kehte hain.

**Zyada math, kam aana-jaana.** Badi matrix multiply: poori matrix ek saath threads mein nahi aati. HBM se ek chhota tile SRAM pe aata hai, aur us tile ka har number kai rows aur columns mein doobara use hota hai. Ek baar lana, phir usi number pe bahut saari multiplies. Cores vyast rehte hain. Is ko **compute-bound** kehte hain.

Activation functions, element-wise ops, aur chhoti layers aksar memory-bound hoti hain. Badi dense matmul aur badi convolution compute-bound ho sakti hain, agar data SRAM mein reuse ho.

## LLM pe yeh kyun matter karta hai

Training ya inference mein weights HBM mein rehte hain. Har token pe model un weights ko padhta hai.

Decode (ek ek token nikalna) mein har weight pe bahut kam math hoti hai, kyunki ek time pe ek hi token ka vector un weights se milta hai. GPU ka time HBM se weights padhne mein nikal jaata hai. Isliye bade inference engines sabse pehle memory bandwidth optimize karte hain, sirf "aur FLOPs" karne ko nahi.

Prefill (poora prompt ek saath) mein ek hi weight kai tokens pe use hota hai, to math zyada hoti hai aur yeh zyada compute-bound hota hai. Yeh farq Day 5 ke aas-paas aur inference week mein wapas aayega. Abhi itna kaafi hai: **jitni baar ek byte reuse ho, utna GPU HBM ke intezaar se baahar nikalta hai.**

SRAM ka kaam yahi hai. Jo numbers agle kuch steps mein baar baar chahiye, unhe HBM se ek baar laao aur SM ke andar rakh lo. Kaise rakhte hain, woh Week 2 ka tiling hai. Aaj sirf yeh dekhna hai ki hierarchy isliye hai.

## Aaj ka picture

```text
CPU DRAM  --PCIe, slow-->  GPU HBM  --slow-->  SM ke andar SRAM  -->  registers  -->  CUDA / Tensor cores
warehouse bahar shehar    GPU ka warehouse     floor ki table         private notebook     workers
```

Workers table se kaam karte hain. Table chhoti hai. Warehouse bada hai. Shehar ke bahar ka warehouse aur bhi door hai.
