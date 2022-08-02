import jax
import jax.numpy as jnp

from functools import partial

# Populate with powers of 2 128 and up
bucket_sizes = []
for s in range(7,31):
    bucket_sizes.append(2**s)
def get_next_bucket_size(s):
    for b in bucket_sizes:
        if s <= b:
            return b
    raise ValueError("max bucket size exceeded")

@partial(jax.jit, static_argnames=("bucket_size"))
def compactify_and_rebucket_arrays(mask, bucket_size, *arrs):
    N_in = mask.sum()
    out_mask = jnp.arange(0, bucket_size) < N_in
    INVALID_IND = bucket_size + 1
    target_inds = jnp.nonzero(mask, size=bucket_size, fill_value=INVALID_IND)

    out_arrs = []
    for a in arrs:
        if a is None:
            out_arrs.append(a)
            continue

        out = a.at[target_inds,...].get(mode='drop').squeeze(0)
        out_arrs.append(out)

    return out_mask, N_in, *out_arrs
     

def fits_in_smaller_bucket(size, curr_bucket_size):
    return get_next_bucket_size(size) < curr_bucket_size
