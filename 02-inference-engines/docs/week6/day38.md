# Day 38: Slurm aur Kubernetes

Dono yeh sawal solve karte hain: is job ko kaun sa node mile.

**Slurm** HPC ka scheduler hai. Tum `sbatch` se job bhejte ho. Woh nodes allocate karta hai. Process seedha metal pe chalti hai. Queue, priority, fair share. Allocation khatam, nodes wapas.

**Kubernetes** containers ka control plane hai. Pod, Service, Deployment. GPU **device plugin** se dikhti hai (Week 7). Job ek custom resource hoti hai. Kueue aur Volcano baad mein aate hain.

Slurm ek allocation deta hai aur khatam ho jaata hai. Kubernetes continuously reconcile karta hai: pod mara, naya banao, desired state ke against.

**Rule:** Slurm ek allocation hai. Kubernetes ek loop hai jo desired state ko sach banata rehta hai.
