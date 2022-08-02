from functools import partial
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp

import utils
from utils import printarr
import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE

# === Function wrappers

class SlopeIntervalImplicitFunction(implicit_function.ImplicitFunction):

    def __init__(self, slope_interval_func):
        super().__init__("classify-and-distance")
        self.slope_interval_func = slope_interval_func

    def __call__(self, params, x):
        return wrap_scalar(partial(self.slope_interval_func, params))(x)

    # the parent class automatically delegates to this
    # def classify_box(self, params, box_lower, box_upper):
        # pass
        
    def classify_general_box(self, params, box_center, box_vecs, offset=0.):

        d = box_center.shape[-1]
        v = box_vecs.shape[-2]
        assert box_center.shape == (d,), "bad box_vecs shape"
        assert box_vecs.shape == (v,d), "bad box_vecs shape"

        # evaluate the function
        input = coordinates_in_general_box(box_center, box_vecs)
        output = self.slope_interval_func(params, input)

        # compute relevant bounds
        slope_lower, slope_upper = slope_bounds(output)
        may_lower, may_upper = primal_may_contain_bounds(output, slope_lower, slope_upper)

        # determine the type of the region
        output_type = SIGN_UNKNOWN
        output_type = jnp.where(may_lower >  offset, SIGN_POSITIVE, output_type)
        output_type = jnp.where(may_upper < -offset, SIGN_NEGATIVE, output_type)
        # output_type = jnp.where(jnp.logical_and(must_lower < 0, must_upper > 0), SIGN_STRADDLES, output_type)
        

        return output_type
    
    def min_distance_to_zero(self, params, box_center, box_axis_vec, return_source_value=False):

        # evaluate the function
        input = coordinates_in_box(box_center-box_axis_vec, box_center+box_axis_vec)
        output = self.slope_interval_func(params, input)
        raw_primal, _, _ = output
        
        # compute relevant bounds
        slope_lower, slope_upper = slope_bounds(output)

        # flip the sign, so we have only one case to handle (where the function is positive)
        # for the slope components, this sign won't matter once we take the abs/max below
        mask = raw_primal >= 0
        primal = jnp.where(mask, raw_primal, -raw_primal)

        # compute the distance to the crossing
        decrease_mag = jnp.maximum(jnp.abs(slope_lower), jnp.abs(slope_upper))
        vec_len = jnp.abs(box_axis_vec) # we're axis-aligned here, so this is just a list of components
        min_len = jnp.min(vec_len)
        decrease_mag = decrease_mag / vec_len # rescale out of the logical [-1,1] domain to wold coords
        axis_decrease = jnp.sum(jnp.clip(decrease_mag, a_min=0.), axis=-1)
        distance_to_zero = jnp.clip(primal / axis_decrease, a_max=min_len)
        distance_to_zero = jnp.where(distance_to_zero == 0, 0., distance_to_zero) # avoid NaN
       
        if return_source_value:
            return raw_primal, distance_to_zero
        else:
            return distance_to_zero
        

    def min_distance_to_zero_in_direction(self, params, source_point, bound_vec, source_range=None, return_source_value=False):

        # construct bounds
        # the "forward" direction always comes first
        # get the center point of the interval
        bound_forward = bound_vec * 0.5
        center = source_point + bound_forward
        bound_forward = bound_forward[None,:] # append a dimension to make it (1,d)
        if source_range is None:
            bound_vecs = bound_forward
        else:
            bound_vecs = jnp.concatenate((bound_forward, source_range))

        # evaluate the function for bounds
        input = coordinates_in_general_box(center, bound_vecs)
        output = self.slope_interval_func(params, input)

        slope_lower, slope_upper = slope_bounds(output)

        if source_range is not None:

            # do another interval evaluation to bound the value at the source along the other vecs
            # alternative: could use the function value from a single evaluation then re-use the slope bounds we already have. This requires one less interval function evaluation, but will give looser bounds.
            input_source = coordinates_in_general_box(source_point, source_range)
            output_source = self.slope_interval_func(params, input_source)
            source_slope_lower, source_slope_upper = slope_bounds(output_source)
            source_lower, source_upper = primal_may_contain_bounds(output_source, source_slope_lower, source_slope_upper)

            # unify the cases based on whether the function is positive or negative, so we only need to handle one case
            # (note that if the interval contains zero this does nonsense, we handle that below)
            is_pos = source_lower >= 0.
            bound_vec_len = jnp.linalg.norm(bound_vec)
            val = jnp.where(is_pos, source_lower, -source_upper) 
            slope = jnp.where(is_pos, slope_lower[0], -slope_upper[0])

            # compute the distance
            slope = 2. * slope / bound_vec_len # remap slope from abstract [-1,1] domain to world
            biggest_decrease = jnp.clip(-slope, a_min=0.)
            distance_to_zero = val / biggest_decrease
            distance_to_zero = jnp.clip(distance_to_zero, a_max=bound_vec_len)
            
            # if the source could possibly be zero, we can't make any progress
            source_contains_zero = jnp.logical_and(source_lower <= 0, source_upper >= 0)
            distance_to_zero = jnp.where(source_contains_zero, 0., distance_to_zero)

          
            if return_source_value:
                return source_lower, source_upper, distance_to_zero
            else:
                return distance_to_zero

        else: # source range is None, this is just a ray

            # evaluate the function at the source
            source_val = self(params, source_point)

            # unify the cases based on whether the function is positive or negative, so we only need to handle one case
            is_pos = source_val >= 0.
            bound_vec_len = jnp.linalg.norm(bound_vec)
            val = jnp.abs(source_val)
            slope = jnp.where(is_pos, slope_lower[0], -slope_upper[0])
             
            # compute the distance
            slope = 2. * slope / bound_vec_len # remap slope from abstract [-1,1] domain to world
            biggest_decrease = jnp.clip(-slope, a_min=0.)
            distance_to_zero = val / biggest_decrease
            distance_to_zero = jnp.clip(distance_to_zero, a_max=bound_vec_len)
           
            # avoid a NaN
            distance_to_zero = jnp.where(source_val==0., 0., distance_to_zero)

            # simple ray case
            if return_source_value:
                return source_val, distance_to_zero
            else:
                return distance_to_zero



# === Slope interval utilities

def is_const(input):
    primal, slope_center, slope_width = input
    return slope_center is None

# Compute the 'radius' (width of the approximation)
# NOTE: this is still in logical [-1,+1] coords
def slope_radius(input):
    if is_const(input): return 0.
    primal, slope_center, slope_width = input
    return slope_width

# Constuct affine inputs for the coordinates in k-dimensional box
# lower,upper should be vectors of length-k
def coordinates_in_box(lower, upper):
    center = 0.5 * (lower+upper)
    vec = upper - center
    axis_vecs = jnp.diag(vec)
    return coordinates_in_general_box(center, axis_vecs)


# Constuct affine inputs for the coordinates in k-dimensional box,
# which is not necessarily axis-aligned
#  - center is the center of the box
#  - vecs is a length-D list of vectors which point from the center of the box to its
#    edges. 
# Note that this is where we remap derivatives to the logical domain [-1,+1]
# (this function is nearly a no-op, but giving it this name makes it easier to
#  reason about)
def coordinates_in_general_box(center, vecs):
    primal = center
    assert center.shape[-1] == vecs.shape[-1], "vecs last dim should be same as center"
    slope_center = jnp.stack(vecs, axis=0)
    slope_width = jnp.zeros_like(slope_center)
    return primal, slope_center, slope_width


def slope_bounds(input):
    primal, slope_center, slope_width = input
    rad = slope_radius(input)
    return slope_center-rad, slope_center+rad

def primal_may_contain_bounds(input, slope_lower, slope_upper):
    primal, _, _ = input
    slope_max_mag = utils.biggest_magnitude(slope_lower, slope_upper)
    primal_rad = jnp.sum(slope_max_mag, axis=0)
    primal_lower, primal_upper = primal-primal_rad, primal+primal_rad 
    return primal_lower, primal_upper


# Convert to/from the slope interval representation from an ordinary value representing a scalar
def from_scalar(x):
    return x, None, None
def to_scalar(input):
    if not is_const(input):
        raise ValueError("non const input")
    return input[0]
def wrap_scalar(func):
    return lambda x : to_scalar(func(from_scalar(x)))
