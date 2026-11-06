# Recovered original development split recipe

This is post-cutoff publication evidence, not a change to the frozen experiment. The coordinator recovered the exact index-construction code from the original preparation tool call after initial metadata-only reconstruction attempts did not match. The code below is that recovered source. It consumes the canonical public `y_train` labels in original order under NumPy 2.2.6; image data are not needed to verify the work and validation indices.

The stable remainder ordering, multiplication order and every RNG call matter. The initial two discarded permutations advance RNG state; their memberships are not returned, displayed, serialized or evaluated. The public command returns aggregate index counts and checksums for work and validation only. It never opens a prepared reserve file.

```python
rng=np.random.default_rng(2026110501)
def allocation(counts,total):
 ideal=counts*(total/counts.sum());out=np.floor(ideal).astype(int)
 for k in np.argsort(-(ideal-out),kind='stable')[:total-out.sum()]:out[k]+=1
 return out
counts=np.bincount(y,minlength=10);reserve_counts=allocation(counts,20000);r1_counts=allocation(reserve_counts,10000)
r1=[];r2=[];dev=[]
for label in range(10):
 ids=rng.permutation(np.flatnonzero(y==label))
 r1.extend(ids[:r1_counts[label]]);r2.extend(ids[r1_counts[label]:reserve_counts[label]]);dev.extend(ids[reserve_counts[label]:])
r1=rng.permutation(r1);r2=rng.permutation(r2);dev=rng.permutation(dev)
vcounts=allocation(np.bincount(y[dev],minlength=10),10000);work=[];validation=[]
for label in range(10):
 ids=rng.permutation(dev[y[dev]==label]);validation.extend(ids[:vcounts[label]]);work.extend(ids[vcounts[label]:])
work=rng.permutation(work);validation=rng.permutation(validation)
```

The installed public `replay.py split --download` command independently reproduced both original work/validation index SHA-256 declarations. The earlier negative reconstruction records remain alongside this recovery; they were index-metadata checks, not model-parameter searches or new accuracy evaluations.
