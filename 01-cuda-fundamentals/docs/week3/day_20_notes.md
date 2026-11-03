# 📅 Day 20-21: Hands-on Pipeline Parallelism & Toy Implementation

Aaj ke day ka target ek functional **Pipeline Parallel (PP)** execution simulation build karna hai, jisme sequential layer execution, Point-to-Point (P2P) CUDA communication (`dist.send` / `dist.recv`), Micro-batching, aur **1F1B (One-Forward-One-Backward)** schedule ka exact execution flow visual aur code level par clear ho sake.

---

## 1. Pipeline Parallel Execution Architecture

Pipeline Parallelism me hum:
1. Model ki total layers ko **Pipeline Stage GPUs** me divide karte hain.
2. Global Input Batch ko **$M$ Micro-batches** me chop karte hain.
3. Peer-to-Peer (`send`/`recv`) calls ke zariye intermediate activation tensors ko next stage par aur activation gradients ($\frac{\partial L}{\partial a}$) ko previous stage par pass karte hain.

```text
===================================================================================
                          STAGE 0 (GPU 0)               STAGE 1 (GPU 1)
===================================================================================
Forward Pass:
[ Input x_m ] ───────► [ Layer 0 Forward ] ──────(P2P Send)──────► [ Layer 1 Forward ]
                                                                          │
                                                                   Compute Loss L_m
                                                                          │
Backward Pass:                                                            ▼
[ Layer 0 Backward ] ◄───(P2P Recv Grad)─── [ Layer 1 Backward ] ◄── Backward Loss
===================================================================================
```

---

## 2. P2P Communication Primitives: `send` vs `isend` / `irecv`

Standard `dist.send` aur `dist.recv` blocking calls hote hain jo CPU ko lock kar dete hain. High-performance pipeline parallelism **Non-blocking P2P primitives** (`isend` aur `irecv`) use karta hai taaki CUDA kernel execution aur network transfers simultaneous chal sakein.

| Primitive | Type | Behavior | Usage in Pipeline Parallelism |
| :--- | :--- | :--- | :--- |
| `dist.send` | Blocking | Wait until receiver receives packet. | Simple debugging / synchronous testing. |
| `dist.recv` | Blocking | Wait until packet arrives in buffer. | Synchronous micro-batch setup. |
| `dist.isend` | Non-blocking | Returns immediately with a `Work` request object. | Overlapping activation send with next micro-batch compute. |
| `dist.irecv` | Non-blocking | Returns immediately; buffer populates in background. | Pre-fetching upcoming micro-batch activation. |

---

## 3. Step-by-Step 1F1B Toy Implementation in PyTorch

Niche ek 2-Stage Pipeline Parallel simulation code hai jo **PyTorch Distributed (P2P)** use karke 1F1B schedule ko implement karta hai:

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist

# Setup stage sub-modules
class Stage0Module(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class Stage1Module(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)

def run_pipeline(rank, world_size, num_microbatches=4, batch_size=16, hidden_dim=128):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    device = torch.device(f"cpu") # Change to f"cuda:{rank}" for GPU setups
    
    if rank == 0:
        model = Stage0Module(hidden_dim).to(device)
    else:
        model = Stage1Module(hidden_dim).to(device)
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Micro-batch storage buffers
    saved_activations = []
    saved_inputs = []

    micro_shape = (batch_size // num_microbatches, hidden_dim)

    # -------------------------------------------------------------
    # STAGE 0 (GPU 0): Forward and P2P Send to STAGE 1
    # -------------------------------------------------------------
    if rank == 0:
        for m in range(num_microbatches):
            x = torch.randn(micro_shape, device=device)
            saved_inputs.append(x)
            
            act = model(x)
            saved_activations.append(act)
            
            # Send activation tensor to Stage 1 (Rank 1)
            dist.send(tensor=act.detach(), dst=1)
            
        # Backward Phase for Stage 0 (Receive grad from Stage 1)
        for m in range(num_microbatches):
            grad_output = torch.zeros(micro_shape, device=device)
            dist.recv(tensor=grad_output, src=1)
            
            # Backpropagate through saved local activation
            saved_activations[m].backward(grad_output)
            
    # -------------------------------------------------------------
    # STAGE 1 (GPU 1): Receive Activations, Compute Loss, Send Grad Back
    # -------------------------------------------------------------
    elif rank == 1:
        labels = [torch.randn(micro_shape, device=device) for _ in range(num_microbatches)]
        
        for m in range(num_microbatches):
            input_act = torch.zeros(micro_shape, device=device, requires_grad=True)
            dist.recv(tensor=input_act, src=0)
            
            out = model(input_act)
            loss = loss_fn(out, labels[m])
            loss.backward()
            
            # Send activation gradient back to Stage 0
            dist.send(tensor=input_act.grad, dst=0)
            
    optimizer.step()
    optimizer.zero_grad()
    print(f"Rank {rank}: Pipeline Step Finished Successfully.")
    dist.destroy_process_group()

if __name__ == "__main__":
    # Test simulation locally using multiprocess spawning
    import torch.multiprocess as mp
    mp.spawn(run_pipeline, args=(2,), nprocs=2)
```

---

## 4. Key Execution Differences: 1F1B vs GPipe in Memory Footprint

| Metric | GPipe Schedule | 1F1B Schedule | Interleaved 1F1B |
| :--- | :--- | :--- | :--- |
| **Activation Memory Overhead** | $\mathcal{O}(M)$ (Stores activations for all $M$ micro-batches before backward) | $\mathcal{O}(P)$ (Stores activations bounded by number of pipeline stages $P$) | $\mathcal{O}(P / v)$ ($v$ = virtual stages per GPU) |
| **Pipeline Bubble Size** | $\frac{P-1}{M + P - 1}$ | $\frac{P-1}{M + P - 1}$ | $\frac{P-1}{v \cdot M + P - 1}$ |
| **P2P Communication Frequency** | High (Batched at step transitions) | Interleaved (Regular send/recv ping-pong) | Highest (Multiple sub-layer transfers) |

---

## 5. Day 20 Summary Checkpoints

1. **Activation Stashing:** Pipeline stages must retain intermediate activation tensors in memory until their corresponding backward step arrives.
2. **Gradient Backward Routing:** Stage $k$ sends $\frac{\partial L}{\partial a_k}$ backward to Stage $k-1$ using P2P operations.
3. **P2P Deadlock Risk:** Blocking `send`/`recv` calls must be strictly paired in opposite order between ranks (Rank 0 Send $\rightarrow$ Rank 1 Recv) to avoid permanent execution hangs.