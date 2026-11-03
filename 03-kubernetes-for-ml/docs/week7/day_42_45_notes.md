# 📅 Day 42-45: GPU Virtualization & Resource Management in Kubernetes

Aapka **Week 7 & 8: Deep Dive into Kubernetes for ML** module ka core foundation yeh section hai. Aaj ke is comprehensive session mein hum yeh samajhenge ki Kubernetes default mein GPUs ko "opaque binary resources" (`nvidia.com/gpu: "1"`) ki tarah treat karta hai—matlab k8s ko ye nahi pata hota ki GPU ke andar SMs (Streaming Multiprocessors), HBM memory, ya NVLink topology kaise kaam kar rahi hai. Is gap ko bridge karne ke liye poori stack niche diye gaye layers par kaam karti hai.

---

## 1. NVIDIA Container Toolkit & Container Device Interface (CDI)

Standard Docker ya Containerd containers run karte waqt isolated namespaces mein hote hain, jinke paas host ke raw hardware accelerators (GPUs, TPUs, ya InfiniBand HCAs) tak direct visibility nahi hoti.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Standard Container vs. GPU Pod                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Standard Pod Container:                                                    │
│  [ Container Namespace ] ──► Cannot see `/dev/nvidia*` or CUDA driver libs  │
│                                                                             │
│  GPU-Enabled Pod Container (NVIDIA Container Toolkit / CDI):                │
│  [ Container Namespace ] ──► Injected Driver Libs + Device Nodes Bind-Mounted │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The CDI & OCI Hook Architecture:
* **Pre-CDI Era:** Pehle ke versions mein humein `--privileged` mode ya volume mounts ke zariye host ke `/dev/nvidia0` ko container mein map karna padta tha, jo security vulnerability create karta tha.
* **Modern CDI Approach:** NVIDIA Container Toolkit ab **Container Device Interface (CDI)** specifications use karta hai. Isme `containerd` ya `CRIO` ko OCI (Open Container Initiative) hook ke through pata chal jata hai ki container start hone se pehle konsi device files aur libraries ko container namespace mein inject karna hai. Isse pod ko bina `privileged: true` mode ke secure tarike se raw GPU access mil jata hai.
* **Injected Components:**
  1. **CUDA Runtime Libraries:** `libcuda.so`, `libcudart.so`, etc., host se container ke rootfs mein mount hoti hain.
  2. **Device Nodes:** `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm` files bind-mount hoti hain.
  3. **Environment Variables:** Automatic injection of `CUDA_VISIBLE_DEVICES` mapping physical card indices to logical container ranks.

---

## 2. Kubernetes Device Plugins Framework

Kubernetes core scheduler apne node par hardware resources track karne ke liye **Device Plugin API** (gRPC-based) use karta hai. Kubelet ko ye nahi pata ki GPU kya hota hai, isiliye NVIDIA ek separate DaemonSet (`nvidia-device-plugin`) run karta hai.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Kubernetes Device Plugin Workflow                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. NVIDIA Device Plugin DaemonSet runs on every K8s Worker Node.           │
│  2. Plugin discovers GPUs via NVML and registers with Kubelet via gRPC:     │
│     `Node.Allocatable["nvidia.com/gpu"] = 8`                                │
│  3. User submits Pod requesting `resources.limits: nvidia.com/gpu: 2`        │
│  4. Kubelet assigns specific GPU IDs and invokes NVIDIA Container Runtime.  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Workflow:
1. **Discovery & Registration:** `nvidia-device-plugin` node par start hokar host NVML (NVIDIA Management Library) se query karta hai ki kitne GPUs aur konsi UUIDs available hain. Yeh gRPC server `/var/lib/kubelet/device-plugins/nvidia.sock` par register hota hai.
2. **Advertising Capacity:** Plugin Kubelet ko batata hai ki is node par kitne hardware units available hain (e.g., `nvidia.com/gpu: 8`).
3. **Allocation Request:** Jab koi Pod `resources.limits: nvidia.com/gpu: 1` ke sath schedule hota hai, tab Kubelet plugin ko `Allocate` gRPC call bhejta hai. Plugin specific GPU IDs (jaise `GPU-a1b2c3...`) return karta hai, jise Kubelet container runtime configuration mein pass kar deta hai.

---

## 3. GPU Virtualization Techniques: MIG vs. MPS vs. Time-Slicing

Jab multiple smaller inference workloads, development notebooks, ya evaluation pipelines ek hi physical GPU ko share karna chahte hain, tab aapko virtualization techniques ki zaroorat padti hai. In teeno ka deep structural tradeoff niche diya gaya hai:

| Feature / Metric | Time-Slicing (Time-Multiplexing) | NVIDIA MPS (Multi-Process Service) | MIG (Multi-Instance GPU) |
| :--- | :--- | :--- | :--- |
| **Isolation Level** | Software-level scheduling | Process-level context sharing | **Hardware-level (Silicon slicing)** |
| **Fault Isolation** | Poor (ek pod crash hua toh sab par asar pad sakta hai) | Poor (CUDA context shared across apps) | **Strong (ek instance fail hone par doosra safe rehta hai)** |
| **Memory Protection** | Shared VRAM (OOM risk across pods) | Shared VRAM pool | **Strict memory slicing (dedicated HBM per slice)** |
| **Best Used For** | Bursty inference, dev/test environments | High-concurrency small model inference | Multi-tenancy, strict SLA enterprise serving |
| **Supported GPUs** | All NVIDIA GPUs (T4, L4, V100, A100, H100) | Volta, Turing, Ampere, Hopper | **Ampere (A100/A30) & Hopper (H100)** onwards |

---

## 4. Deep Dive: MIG (Multi-Instance GPU) Architecture

A100 ya H100 jaise enterprise GPUs mein **MIG** hardware level par ek GPU ko upar se lekar niche tak **completely isolated physical instances** mein partition kar deta hai.

```
┌─────────────────────────────────────────────────────────────────────────────┤
│                   NVIDIA A100 / H100 MIG Partitions                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Physical GPU (e.g., 80GB HBM3 A100)                                        │
│  ┌───────────────────────┬───────────────────────┬───────────────────────┐  │
│  │ MIG 1g.10gb           │ MIG 2g.20gb           │ MIG 4g.40gb           │  │
│  │ (1 SM Slice, 10GB HBM)│ (2 SM Slices, 20GB HBM│ (4 SM Slices, 40GB HBM│  │
│  │ Isolated Cache & Mem  │ Isolated Cache & Mem  │ Isolated Cache & Mem  │  │
│  └───────────────────────┴───────────────────────┴───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Slicing Mechanics & QoS:
* **Partition Profiles:** Aap ek 80GB A100 ko alag-alag flexible profiles mein tod sakte hain:
  * `1g.10gb` (1 Streaming Multiprocessor slice, 10GB dedicated HBM)
  * `2g.20gb` (2 SM slices, 20GB HBM)
  * `3g.40gb`, `4g.40gb`, `7g.80gb` (Full GPU)
* **Hardware Isolation:** Har MIG instance ke paas apne dedicated SMs, dedicated memory controllers, crossbar paths, aur L2 cache partitions hote hain. Ek instance ka heavy memory read/write doosre instance ke bandwidth ya compute latency ko affect nahi kar sakta. Kubernetes mein ise expose karne ke liye `nvidia.com/mig-1g.10gb` jese extended resources use kiye jaate hain.

---

## 5. Fractional GPU Sharing via Time-Slicing & MPS

Agar aapke paas older GPUs hain (jaise T4, L4, V100) jo MIG hardware slicing support nahi karte, tab hum software-based sharing use karte hain:

### A. NVIDIA Multi-Process Service (MPS)
* Allows multiple CUDA contexts from different client processes to share a single GPU simultaneously.
* Hardware-level task scheduling multiplexes kernels onto the GPU Streaming Multiprocessors, significantly reducing context-switch overhead compared to basic time-slicing.

### B. Kubernetes Time-Slicing Device Plugins
* Kubelet ko over-subscribe karne ke liye config di jati hai jisme ek physical GPU ko multiple virtual GPUs (e.g., 4 replicas) mein map kiya jata hai.
* **The Catch:** Isme real hardware partitioning nahi hoti. Kubelet ko lagta hai ki node par 4 GPUs hain, lekin asal mein ek hi GPU par contexts rapidly switch hote hain (Time-Multiplexing). Agar dono pods ek sath heavy matrix multiplication run karein, toh latency spike ho jati hai aur memory overcommit hone par OOM Killer active ho jata hai.

---

## 6. Summary Checkpoints for Days 42-45

1. **NVIDIA Container Toolkit (CDI):** Bridges host GPU drivers and container namespaces securely via OCI hooks without needing `privileged: true`.
2. **Device Plugins:** Enables Kubernetes Kubelet to discover, register, and allocate custom hardware resources like `nvidia.com/gpu`.
3. **MIG vs. Fractional Sharing:** MIG provides hardware-isolated physical slices with guaranteed Quality of Service (QoS), while MPS and Time-Slicing provide software multiplexing for density optimization on legacy cards.