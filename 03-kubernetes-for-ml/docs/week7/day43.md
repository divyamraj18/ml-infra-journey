# Day 43: Device plugin

Kubelet ko pata nahi hota ki node pe 8 GPUs hain, jab tak koi use bataye.

**Device plugin** yeh count advertise karta hai. Pod yeh maangta hai:

```text
resources:
  nvidia.com/gpu: 1
```

Scheduler ek GPU choose karta hai. Plugin `NVIDIA_VISIBLE_DEVICES` set karta hai, taaki process sirf wahi GPU dekhe. Doosri GPUs us container mein gayab rehti hain.

**Rule:** device plugin count deta hai aur ek pod ko ek GPU dikhata hai.
