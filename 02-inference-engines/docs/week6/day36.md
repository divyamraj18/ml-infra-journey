# Day 36: Elastic training

Fixed world size: ek node gaya, job mar gayi. Ring toot gayi.

**TorchElastic** world size badal sakta hai. Node wapas aaye, ranks phir se join karein, checkpoint se aage badhein. Job ko shuru se nahi daudna.

Yeh magic nahi hai. Checkpoint consistent hona chahiye. Worker count badle to batch size aur learning rate ke rules pehle se likhe hone chahiye. Warna wahi loss curve do alag training ban jaati hai.

**Rule:** elastic ka matlab naya worker join kar sake. Iska price consistent checkpoint hai.
