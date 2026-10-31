# Day 57: Ray kyun

Ek Python function likhna aasaan hai. Use 20 machines pe chalana mushkil hai: SSH, processes, fail hua to kaun dobara chalaye, data kahan pada hai.

**Ray** yeh gap bharta hai. Function pe decorator lagao. Ray use cluster pe chalaata hai.

Yeh NCCL ki jagah nahi hai. NCCL GPU bytes move karta hai. Ray tasks aur actors schedule karta hai. Dono ek job mein saath reh sakte hain: Ray worker ke andar NCCL.

**Rule:** Ray processes aur data ka control plane hai. NCCL GPU ka data plane hai.
