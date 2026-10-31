# Day 37: Health checks, fencing, silent NaN

Node "up" hai aur phir bhi loss `NaN` hai. Kabhi ek GPU ki memory kharab hoti hai. Gradient mein kuch values inf ho jaati hain. AllReduce poore cluster ko NaN baant deta hai. Crash nahi hota. Training quietly kharab ho jaati hai.

**Health check**, job se pehle: GPU compute test, NIC up, disk writable.

**Fence:** bure node ko scheduler se hatao, taaki agla job usi pe na lage. Kubernetes mein isko cordon kehte hain.

Training ke dauran NaN dikhe to us rank ko maro, elastic restart karo, bad node ko fence karo. NaN ko AllReduce mein mat ghusne do.

**Rule:** crash se pehle wala bug silent NaN hai. Use cluster mein mat baantne do.
