# Day 46: Gang scheduling

Default Kubernetes scheduler ek pod ko ek node pe rakh deta hai. Training ko **saare** workers chahiye. Ek rank missing ho to NCCL ka ring atak jaata hai (Day 35).

**Gang scheduling.** Sab pods schedule hon, tab koi start ho. Ek bhi pending ho to jo chal rahe the unhe bhi mat chalao.

Warna 7 pods GPUs pakad kar 8th ka wait karte hain. Doosri job woh GPUs use nahi kar sakti. Isko partial-start deadlock kehte hain.

**Rule:** training ka start "sab ya koi nahi" hai.
