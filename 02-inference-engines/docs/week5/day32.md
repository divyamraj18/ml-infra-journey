# Day 32: Checkpoint time aur uptime

Step 2 second ka hai. Checkpoint 3 minute ka hai, har 30 minute. Un 3 minutes mein GPUs idle hain. 1024 GPUs pe yeh bahut mehnga idle hai.

**Sync checkpoint.** Likh ke khatam, phir agla step. Crash pe data safe hai. Step time lamba hai.

**Async checkpoint.** Agla step chalu, copy background mein disk pe jaaye. Crash us window mein ho to last good checkpoint purana hai. Kuch steps dobara karne padenge.

Kitni der baad checkpoint likhte ho, utna kaam dobara karne ko taiyar raho. Is window ko **recovery point** socho. Har step likhna safe hai aur cluster ko maar deta hai.

**Rule:** checkpoint frequency = kitna kaam dobara kar sakte ho, uske against kitna GPU-time likhne mein jaayega.
