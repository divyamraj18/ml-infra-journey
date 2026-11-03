# 📅 Day 17: Hands-on PyTorch DDP Scripting & Local Multi-Process Simulation

Aaj ke day ka goal ek production-ready **PyTorch Distributed Data Parallel (DDP)** Python script ready karna hai aur use locally multiple processes (`torchrun`) launch karke run karna hai.

---

## 1. DDP Script Architecture & Lifecycle

Local multi-process simulation execute karne ke liye script me basically 5 core steps setup karne padte hain:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DDP Script Execution Steps                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
 1. Environment Parsing ────► Parse RANK, WORLD_SIZE, and LOCAL_RANK from env.
                                       │
                                       ▼
 2. Process Group Init  ────► Initialize NCCL backend (`dist.init_process_group`).
                                       │
                                       ▼
 3. Model Wrapping      ────► Move model to `LOCAL_RANK` and wrap with `DDP()`.
                                       │
                                       ▼
 4. Distributed Sampler ────► Wrap DataLoader with `DistributedSampler`.
                                       │
                                       ▼
 5. Distributed Loop    ────► Set epoch index on sampler and run train loop.
```

---

## 2. Hands-on PyTorch DDP Code Implementation

Yahan complete runnable script di gayi hai jise aap single-machine multi-GPU ya simulated multi-process environment par execute kar sakte hain:

```python
# train_ddp.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. Dummy Dataset setup
class SyntheticDataset(Dataset):
    def __init__(self, size=1000, features=4096):
        self.data = torch.randn(size, features)
        self.labels = torch.randn(size, 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def setup_ddp():
    """Environment variables setup by torchrun parsing"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Device binding per process
    torch.cuda.set_device(local_rank)

    # Initialize NCCL process group
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    rank, world_size, local_rank = setup_ddp()

    # 2. Instantiate Model and Transfer to Target Local GPU
    model = nn.Sequential(
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 10)
    ).to(local_rank)

    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[local_rank])

    # 3. Setup Distributed Sampler
    # Ensures each rank gets a unique, non-overlapping subset of samples per epoch
    dataset = SyntheticDataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    # 4. Training Loop
    epochs = 3
    for epoch in range(epochs):
        # Mandatory: Set epoch on sampler so shuffling differs each epoch!
        sampler.set_epoch(epoch)
        ddp_model.train()

        total_loss = 0.0
        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(local_rank, non_blocking=True)
            targets = targets.to(local_rank, non_blocking=True)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, targets)

            # Gradient computation & NCCL AllReduce overlap triggered here
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} | Average Loss: {total_loss / len(dataloader):.4f}")

    cleanup()

if __name__ == "__main__":
    main()
```

---

## 3. Launching DDP via `torchrun`

PyTorch me `torchrun` script ko launch karne and environment variables inject karne ka canonical utility tool hai.

### Local Execution Command (Simulating 4 GPUs/Processes):

```bash
# Set NCCL logging to verify communication backend initialization
export NCCL_DEBUG=INFO

# Launch script on 4 local GPUs (or simulated CPU/GPU processes)
torchrun --standalone --nproc_per_node=4 train_ddp.py
```

### Environment Variables Injected by `torchrun`:

| Variable | Description |
| --- | --- |
| `RANK` | Global process ID across all nodes and GPUs ($0, 1, \dots, N-1$). |
| `WORLD_SIZE` | Total number of processes running across all nodes. |
| `LOCAL_RANK` | Device ID within the current physical node ($0, 1, \dots, \text{GPUs per node}-1$). |
| `MASTER_ADDR` | IP address of the Rank 0 node for process group initialization. |
| `MASTER_PORT` | Free TCP port on the Master node. |

---

## 4. Key Takeaways for Day 17

1. **`DistributedSampler` is Mandatory:** Without `DistributedSampler`, every rank processes the *exact same* dataset batches, wasting compute and breaking optimization dynamics.
2. **`sampler.set_epoch(epoch)`:** Always update the epoch index before iterating through the loader; otherwise, data shuffling order remains static across epochs.
3. **`torchrun` Automation:** Avoid legacy `torch.multiprocessing.spawn` manually. `torchrun` manages rank assignment, process monitoring, and dynamic fault tolerance automatically.