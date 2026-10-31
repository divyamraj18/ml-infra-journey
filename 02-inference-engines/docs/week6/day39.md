# Day 39: Slurm kab, Kubernetes kab

| | Slurm | Kubernetes |
|---|---|---|
| History | HPC, bare metal | Cloud, containers |
| Speed | Kam layers. Process seedha GPU pe. | Container, CNI, extra hops. |
| Gang | Native: saari nodes milen tab job shuru. | Extra: Volcano ya Kueue. Week 7–8. |
| Serving | Kamzor fit. | Services aur autoscaling natural hain. |
| Training | Bahut bade clusters isi pe bade hue. | Ab common hai. Gang aur NCCL network extra kaam hain. |

Meta aur OpenAI ne dono ke ideas mix kiye hain: Slurm jaisa gang aur priority, aur container images.

**Rule:** training ko gang chahiye. Serving ko service chahiye. Tool us hisaab se chuno.
