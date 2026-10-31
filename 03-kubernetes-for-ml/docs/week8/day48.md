# Day 48: Kueue

**Kueue** queues, quotas, aur fair sharing lagata hai. Research team A ko 16 GPUs, team B ko 8. Job admit tab hoti hai jab quota bhi ho aur gang bhi ban sake.

**Flavor** GPU type hai. A100 queue aur H100 queue alag ho sakte hain. Galat flavor pe job baith jaaye to NCCL ya memory assumption toot jaati hai.

Multi-cluster baad ka topic hai. Pehle ek cluster pe queue samajh lo: admit, quota, gang.

**Rule:** Kueue "kaun si team, kaun si GPU, kitni" decide karta hai. Volcano jaisa gang us admit ke andar hai.
