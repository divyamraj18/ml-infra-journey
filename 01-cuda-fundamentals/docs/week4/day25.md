# Day 25: CPU aur NVMe offload

ZeRO-3 ke baad bhi GPU chhoti ho to state GPU se bahar rakho.

**Optimizer offload.** `m` aur `v` CPU RAM pe rehte hain. Step CPU pe lagta hai. Updated weights wapas GPU pe aate hain. PCIe slow hai (Day 9). Step time badhta hai. GPU memory mein Adam ki diary nahi baithti.

**Parameter offload, ZeRO-Infinity.** Weights bhi CPU ya NVMe pe. Layer chahiye tab GPU pe lao, use karo, wapas bhejo. NVMe terabytes deta hai. Bandwidth PCIe ya NVMe se cap hoti hai. GPU aksar data ka wait karti hai.

Yeh tab lo jab warna train ho hi na sake. 8× H100 mil jayein to offload ki jagah ZeRO-3 ya tensor parallel sasta padta hai.

**Rule:** offload memory CPU/NVMe se kharidta hai aur step time PCIe ko de deta hai.
