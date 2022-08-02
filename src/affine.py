from functools import partial
import dataclasses 
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp

import utils

import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE

# === Function wrappers

class AffineImplicitFunction(implicit_function.ImplicitFunction):

    def __init__(self, affine_func, ctx):
        super().__init__("classify-only")
        self.affine_func = affine_func
        self.ctx = ctx
        self.mode_dict = {'ctx' : self.ctx}


    def __call__(self, params, x):
        f = lambda x : self.affine_func(params, x, self.mode_dict)
        return wrap_scalar(f)(x)

    # the parent class automatically delegates to this
    # def classify_box(self, params, box_lower, box_upper):
        # pass
        
    def classify_general_box(self, params, box_center, box_vecs, offset=0.):

        d = box_center.shape[-1]
        v = box_vecs.shape[-2]
        assert box_center.shape == (d,), "bad box_vecs shape"
        assert box_vecs.shape == (v,d), "bad box_vecs shape"
        keep_ctx = dataclasses.replace(self.ctx, affine_domain_terms=v)

        # evaluate the function
        input = coordinates_in_general_box(keep_ctx, box_center, box_vecs)
        output = self.affine_func(params, input, {'ctx' : keep_ctx})

        # compute relevant bounds
        may_lower, may_upper = may_contain_bounds(keep_ctx, output)
        # must_lower, must_upper = must_contain_bounds(keep_ctx, output)

        # determine the type of the region
        output_type = SIGN_UNKNOWN
        output_type = jnp.where(may_lower >  offset, SIGN_POSITIVE, output_type)
        output_type = jnp.where(may_upper < -offset, SIGN_NEGATIVE, output_type)

        return output_type

# === Affine utilities

# We represent affine data as a tuple input=(base,aff,err). Base is a normal shape (d,) primal vector value, affine is a (v,d) array of affine coefficients (may be v=0), err is a centered interval error shape (d,), which must be nonnegative.
# For constant values, aff == err == None. If is_const(input) == False, then it is guaranteed that aff and err are non-None.

@dataclass(frozen=True)
class AffineContext():
    mode: str = 'affine_fixed'
    truncate_count: int = -777
    truncate_policy: str = 'absolute'
    affine_domain_terms: int = 0
    n_append: int = 0

    def __post_init__(self):
        if self.mode not in ['interval', 'affine_fixed', 'affine_truncate', 'affine_append', 'affine_all']:
            raise ValueError("invalid mode")

        if self.mode == 'affine_truncate':
            if self.truncate_count is None:
                raise ValueError("must specify truncate count")

def is_const(input):
    base, aff, err = input
    if err is not None: return False
    return aff is None or aff.shape[0] == 0


# Compute the 'radius' (width of the approximation)
def radius(input):
    if is_const(input): return 0.
    base, aff, err = input
    rad = jnp.sum(jnp.abs(aff), axis=0)
    if err is not None:
        rad += err
    return rad

# Constuct affine inputs for the coordinates in k-dimensional box
# lower,upper should be vectors of length-k
def coordinates_in_box(ctx, lower, upper):
    center = 0.5 * (lower+upper)
    vec = upper - center
    axis_vecs = jnp.diag(vec)
    return coordinates_in_general_box(ctx, center, axis_vecs)

# Constuct affine inputs for the coordinates in k-dimensional box,
# which is not necessarily axis-aligned
#  - center is the center of the box
#  - vecs is a (V,D) array of vectors which point from the center of the box to its
#    edges. These will correspond to each of the affine symbols, with the direction
#    of the vector becoming the positive orientaiton for the symbol.
# (this function is nearly a no-op, but giving it this name makes it easier to
#  reason about)
def coordinates_in_general_box(ctx, center, vecs):
    base = center
    if ctx.mode == 'interval':
        aff = jnp.zeros((0,center.shape[-1]))
        err = jnp.sum(jnp.abs(vecs), axis=0)
    else:
        aff = vecs
        err = jnp.zeros_like(center)
    return base, aff, err

def may_contain_bounds(ctx, input,):
    '''
    An interval range of values that `input` _may_ take along the domain
    '''
    base, aff, err = input
    rad = radius(input)
    return base-rad, base+rad

def truncate_affine(ctx, input):
    # do nothing if the input is a constant or we are not in truncate mode
    if is_const(input): return input
    if ctx.mode != 'affine_truncate':
        return input

    # gather values
    base, aff, err = input
    n_keep = ctx.truncate_count

    # if the affine list is shorter than the truncation length, nothing to do
    if aff.shape[0] <= n_keep:
        return input

    # compute the magnitudes of each affine value
    # TODO fanicier policies?
    if ctx.truncate_policy == 'absolute':
        affine_mags = jnp.sum(jnp.abs(aff), axis=-1)
    elif ctx.truncate_policy == 'relative':
        affine_mags = jnp.sum(jnp.abs(aff), axis=-1) / jnp.abs(base)
    else:
        raise RuntimeError("bad policy")


    # sort the affine terms by by magnitude
    sort_inds = jnp.argsort(-affine_mags, axis=-1) # sort to decreasing order
    aff = aff[sort_inds,:]

    # keep the n_keep highest-magnitude entries
    aff_keep = aff[:n_keep,:]
    aff_drop = aff[n_keep:,:]

    # for all the entries we aren't keeping, add their contribution to the interval error
    err = err + jnp.sum(jnp.abs(aff_drop), axis=0)

    return base, aff_keep, err

def apply_linear_approx(ctx, input, alpha, beta, delta):
    base, aff, err = input
    base = alpha * base + beta
    if aff is not None:
        aff = alpha * aff

    # This _should_ always be positive by definition. Always be sure your 
    # approximation routines are generating positive delta.
    # At most, we defending against floating point error here.
    delta = jnp.abs(delta)

    if ctx.mode in ['interval', 'affine_fixed']:
        err = alpha * err + delta
    elif ctx.mode in ['affine_truncate', 'affine_all']:
        err = alpha * err
        new_aff = jnp.diag(delta)
        aff = jnp.concatenate((aff, new_aff), axis=0)
        base, aff, err = truncate_affine(ctx, (base, aff, err))

    elif ctx.mode in ['affine_append']:
        err = alpha * err
        
        keep_vals, keep_inds = jax.lax.top_k(delta, ctx.n_append)
        row_inds = jnp.arange(ctx.n_append)
        new_aff = jnp.zeros((ctx.n_append, aff.shape[-1]))
        new_aff = new_aff.at[row_inds, keep_inds].set(keep_vals)
        aff = jnp.concatenate((aff, new_aff), axis=0)
        err = err + (jnp.sum(delta) - jnp.sum(keep_vals)) # add in the error for the affs we didn't keep

    return base, aff, err

# Convert to/from the affine representation from an ordinary value representing a scalar
def from_scalar(x):
    return x, None, None
def to_scalar(input):
    if not is_const(input):
        raise ValueError("non const input")
    return input[0]
def wrap_scalar(func):
    return lambda x : to_scalar(func(from_scalar(x)))
