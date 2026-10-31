# Day 29: Parallel file systems

Checkpoint ek chhoti file nahi hoti. Hazaaron processes ek saath likhte hain. **NFS** ek server pe atak jaata hai. **S3** har chhoti file pe HTTP ka cost lagata hai. "Checkpoint likh rahe hain" GPU idle time ban jaata hai.

**Parallel file system:** Lustre, GPFS (Spectrum Scale), WekaIO. Metadata alag servers. Data alag servers. Kai clients ek saath likhte hain. Checkpoint ka pattern yahi hai: bahut saare writers, badi sequential files.

NFS ya S3 tab theek hain jab ek process dheere se padhe. Training step ke andar ke burst ke liye woh nahi bane.

**Rule:** checkpoint storm ko parallel FS chahiye. Object store final copy ke liye hai, step ke beech ki scratch ke liye nahi.
