## Experiment 003

Still in progress.



`weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))` here masked fill is an out-
of-place operation.  --> Used SDPA.