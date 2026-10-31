# Day 31: DataLoader aur WebDataset

PyTorch `DataLoader` ke workers alag-alag files kholte hain. Karoron chhoti files ka matlab metadata storm hai. GPU batch ka wait karti hai. Yeh bhi stall hai, HBM ka nahi, disk ka.

**WebDataset** samples ko tar **shards** mein rakhta hai. Hazaron samples ek badi file. Read sequential hai. Workers shard padhte hain, random tiny files nahi.

**Rule:** GPU ko batch time pe chahiye. Chhoti files metadata pe atakti hain. Shards sequential hote hain.
