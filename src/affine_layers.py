from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

import affine
import mlp
import utils

def dense(input, A, b, ctx):
    if(affine.is_const(input)):
        out = jnp.dot(input[0], A)
        if b is not None:
            out += b
        return  out, None, None

    base, aff, err = input

    def dot(x, with_abs=False):
        myA = jnp.abs(A) if with_abs else A 
        return jnp.dot(x, myA)
 
    base = dot(base)
    aff = jax.vmap(dot)(aff)
    err = dot(err, with_abs=True)

    if b is not None:
        base += b

    return base, aff, err
mlp.apply_func['affine']['dense'] = dense

def relu(input, ctx):
    # Chebyshev bound
    base, aff, err = input

    if affine.is_const(input):
        return jax.nn.relu(base), aff, err

    lower, upper = affine.may_contain_bounds(ctx, input)

    # Compute the linearized approximation
    alpha = (jax.nn.relu(upper) - jax.nn.relu(lower)) / (upper - lower)
    alpha = jnp.where(lower >= 0, 1., alpha)
    alpha = jnp.where(upper < 0, 0., alpha)
    # handle numerical badness in the denominator above
    alpha = jnp.nan_to_num(alpha, nan=0.0, copy=False) # necessary?
    alpha = jnp.clip(alpha, a_min=0., a_max=1.) 

    # here, alpha/beta are necessarily positive, which makes this simpler
    beta = (jax.nn.relu(lower) - alpha * lower) / 2
    delta = beta

    output = affine.apply_linear_approx(ctx, input, alpha, beta, delta)
    return output
mlp.apply_func['affine']['relu'] = relu

def elu(input, ctx):
    # Chebyshev bound
    # Confusingly, elu has a parameter typically called 'alpha', and we also use 'alpha' for our linearizaiton notation. Here we simply ignore and do not support elu's alpha.
    base, aff, err = input

    if affine.is_const(input):
        return jax.nn.elu(base), aff, err

    lower, upper = affine.may_contain_bounds(ctx, input)

    # Compute the linearized approximation
    lowerF = jax.nn.elu(lower)
    upperF = jax.nn.elu(upper)
    # lowerS = jnp.where(lower < 0, lowerF + 1., 1.)
    # upperS = jnp.where(upper < 0, upperF + 1., 1.)
    lowerS = jnp.minimum(jnp.exp(lower), 1.) # more numerically stable than ^^^, but costs exp()
    upperS = jnp.minimum(jnp.exp(upper), 1.)

    alpha = (upperF - lowerF) / (upper - lower)
    alpha = jnp.where(lower >= 0, 1., alpha)
    # handle numerical badness in the denominator above
    alpha = jnp.nan_to_num(alpha, nan=0.0, copy=False) # necessary?
    alpha = jnp.clip(alpha, a_min=lowerS, a_max=upperS) 

    # the alpha tangent point necessarily occurs in the <= 0. part of the function
    r_upper = (lowerF - alpha * lower)
    x_lower = jnp.clip(jnp.log(alpha), a_min=lower, a_max=upper)
    r_lower = (alpha-1.) - alpha * x_lower
    beta = 0.5 * (r_upper + r_lower)
    # delta = r_upper - beta
    delta = 0.5 * jnp.abs(r_upper - r_lower) # this is very defensive, to ensure delta>=0

    # in strictly > 0 case, just directly set the result
    alpha = jnp.where(lower >= 0, 1., alpha)
    beta = jnp.where(lower >= 0, 0., beta)
    delta = jnp.where(lower >= 0, 0., delta)

    output = affine.apply_linear_approx(ctx, input, alpha, beta, delta)
    return output
mlp.apply_func['affine']['elu'] = elu

def sin(input, ctx):
    # not-quite Chebyshev bound
    base, aff, err = input
    pi = jnp.pi

    if affine.is_const(input):
        return jnp.sin(base), aff, err

    lower, upper = affine.may_contain_bounds(ctx, input)

    slope_lower, slope_upper = utils.cos_bound(lower, upper)
    alpha = 0.5 * (slope_lower + slope_upper) # this is NOT the Chebyshev value, but seems reasonable
    alpha = jnp.clip(alpha, a_min=-1., a_max=1.) # (should already be there, this is for numerics only)

    # We want to find the minima/maxima of (sin(x) - alpha*x) on [lower, upper] to compute our 
    # beta and delta. In addition to the endpoints, some calc show there can be interior 
    # extrema at +-arccos(alpha) + 2kpi for some integer k.
    # The extrema will 
    intA = jnp.arccos(alpha)
    intB = -intA

    # The the first and last occurence of a value which repeats mod 2pi on the domain [lower, upper]
    # (these give the only possible locations for our extrema)
    def first(x): return 2.*pi*jnp.ceil((lower + x) / (2.*pi)) - x
    def last(x): return 2.*pi*jnp.floor((upper - x) / (2.*pi)) + x

    extrema_locs = [lower, upper, first(intA), last(intA), first(intB), last(intB)]
    extrema_locs = [jnp.clip(x, a_min=lower, a_max=upper) for x in extrema_locs]
    extrema_vals = [jnp.sin(x) - alpha * x for x in extrema_locs]

    r_lower = utils.minimum_all(extrema_vals)
    r_upper = utils.maximum_all(extrema_vals)

    beta = 0.5 * (r_upper + r_lower)
    delta = r_upper - beta

    output = affine.apply_linear_approx(ctx, input, alpha, beta, delta)
    return output
mlp.apply_func['affine']['sin'] = sin

def pow2_frequency_encode(input, ctx, coefs, shift=None):
    base, aff, err = input

    # TODO debug
    if len(base.shape) > 1:
        raise ValueError("big base")

    # expand the length-d inputs to a lenght-d*c vector
    def s(with_shift, x): 
        out = (x[:,None] * coefs[None,:])
        if with_shift and shift is not None:
            out += shift
        return out.flatten()

    base = s(True, base)
    if affine.is_const(input):
        return base, None, None

    aff = jax.vmap(partial(s, False))(aff)
    err = s(False, err)
    
    return base, aff, err
mlp.apply_func['affine']['pow2_frequency_encode'] = pow2_frequency_encode

def squeeze_last(input, ctx):
    base, aff, err = input
    s = lambda x : jnp.squeeze(x, axis=0)
    base = s(base)
    if affine.is_const(input):
        return base, None, None
    aff = jax.vmap(s)(aff)
    err = s(err)
    return base, aff, err
mlp.apply_func['affine']['squeeze_last'] = squeeze_last

def spatial_transformation(input, R, t, ctx):
    # if the shape transforms by R,t, input points need the opposite transform
    R_inv = jnp.linalg.inv(R)
    t_inv = jnp.dot(R_inv, -t)
    return dense(input, A=R_inv, b=t_inv, ctx=ctx)
mlp.apply_func['affine']['spatial_transformation'] = spatial_transformation
