from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

import slope_interval
import mlp
import utils

def dense(input, A, b):
    if(slope_interval.is_const(input)):
        out = jnp.dot(input[0], A)
        if b is not None:
            out += b
        return  out, None, None
    
    primal, dual_center, dual_width = input
    assert (len(primal.shape) == 1 and dual_center.shape == dual_width.shape and dual_center.shape[-1] == primal.shape[-1]), "shape error"

    def dot(x):
        return jnp.dot(x, A)
    def dot_abs(x):
        return jnp.dot(x, jnp.abs(A))
 
    primal = dot(primal)
    if b is not None:
        primal = primal + b

    if dual_center is not None: dual_center = jax.vmap(dot)(dual_center)
    if dual_width is not None: dual_width = jax.vmap(dot_abs)(dual_width)

    return primal, dual_center, dual_width
mlp.apply_func['slope_interval']['dense'] = dense

def relu(input):
    primal, dual_center, dual_width = input

    new_primal = jax.nn.relu(primal)

    if slope_interval.is_const(input):
        return new_primal, dual_center, dual_width

    slope_lower, slope_upper = slope_interval.slope_bounds(input)
    primal_lower, primal_upper = slope_interval.primal_may_contain_bounds(input, slope_lower, slope_upper)

    # compute bounds on the derivative of the nonlineariy
    df_lower = jnp.where(primal_lower > 0, 1., 0.)
    df_upper = jnp.where(primal_upper < 0, 0., 1.)

    # simpler here because df is nonegative
    new_slope_lower = jnp.minimum(slope_lower * df_lower[None,:], slope_lower * df_upper[None,:])
    new_slope_upper = jnp.maximum(slope_upper * df_lower[None,:], slope_upper * df_upper[None,:])
    new_slope_center = 0.5 * (new_slope_lower + new_slope_upper)
    new_slope_width = new_slope_upper - new_slope_center

    return new_primal, new_slope_center, new_slope_width
mlp.apply_func['slope_interval']['relu'] = relu

def elu(input):
    primal, dual_center, dual_width = input

    new_primal = jax.nn.elu(primal)

    if slope_interval.is_const(input):
        return new_primal, dual_center, dual_width

    slope_lower, slope_upper = slope_interval.slope_bounds(input)
    primal_lower, primal_upper = slope_interval.primal_may_contain_bounds(input, slope_lower, slope_upper)

    # compute bounds on the derivative of the nonlineariy
    df_lower = jnp.clip(jnp.exp(primal_lower), a_max=1.)
    df_upper = jnp.clip(jnp.exp(primal_upper), a_max=1.)

    # simpler here because df is nonegative
    new_slope_lower = jnp.minimum(slope_lower * df_lower[None,:], slope_lower * df_upper[None,:])
    new_slope_upper = jnp.maximum(slope_upper * df_lower[None,:], slope_upper * df_upper[None,:])
    new_slope_center = 0.5 * (new_slope_lower + new_slope_upper)
    new_slope_width = new_slope_upper - new_slope_center

    return new_primal, new_slope_center, new_slope_width
mlp.apply_func['slope_interval']['elu'] = elu

def sin(input):
    primal, dual_center, dual_width = input

    new_primal = jnp.sin(primal)

    if slope_interval.is_const(input):
        return new_primal, dual_center, dual_width

    slope_lower, slope_upper = slope_interval.slope_bounds(input)
    primal_lower, primal_upper = slope_interval.primal_may_contain_bounds(input, slope_lower, slope_upper)

    # compute bounds on the derivative of the nonlineariy
    df_lower, df_upper = utils.cos_bound(primal_lower, primal_upper)
    # utils.printarr(primal_lower, primal_upper, df_lower, df_upper, short=False)

    # df can be positive or negative; need full expression
    # (this is just an interval multiplication)
    vals = [slope_lower * df_lower[None,:], slope_lower * df_upper[None,:], 
            slope_upper * df_lower[None,:], slope_upper * df_upper[None,:]]
    new_slope_lower = utils.minimum_all(vals)
    new_slope_upper = utils.maximum_all(vals)
    new_slope_center = 0.5 * (new_slope_lower + new_slope_upper)
    new_slope_width = new_slope_upper - new_slope_center

    return new_primal, new_slope_center, new_slope_width
mlp.apply_func['slope_interval']['sin'] = sin

def pow2_frequency_encode(input, coefs, shift=None):
    primal, dual_center, dual_width = input

    # expand the length-d inputs to a lenght-d*c vector
    def s(with_shift, x): 
        out = (x[:,None] * coefs[None,:])
        if with_shift and shift is not None:
            out += shift
        return out.flatten()

    primal = s(True, primal)
    if dual_center is not None: dual_center = jax.vmap(partial(s, False))(dual_center)
    if dual_width is not None: dual_width = jax.vmap(partial(s,False))(dual_width)
    
    return primal, dual_center, dual_width
mlp.apply_func['slope_interval']['pow2_frequency_encode'] = pow2_frequency_encode

def squeeze_last(input):
    primal, dual_center, dual_width = input
    s = lambda x : jnp.squeeze(x, axis=0)
    primal = s(primal)
    if dual_center is not None: 
        dual_center = jax.vmap(s)(dual_center)
        dual_width = jax.vmap(s)(dual_width)
    return primal, dual_center, dual_width
mlp.apply_func['slope_interval']['squeeze_last'] = squeeze_last

def spatial_transformation(input, R, t):
    # if the shape transforms by R,t, input points need the opposite transform
    R_inv = jnp.linalg.inv(R)
    t_inv = jnp.dot(R_inv, -t)
    return dense(input, A=R_inv, b=t_inv)
mlp.apply_func['slope_interval']['spatial_transformation'] = spatial_transformation
