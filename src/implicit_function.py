from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import utils

# "enums" integer codes denoting the sign of the implicit function with a region
SIGN_UNKNOWN = 0    # could be anything
SIGN_POSITIVE = 1   # definitely positive throughout
SIGN_NEGATIVE = 2   # definitely negative throughout

class ImplicitFunction:

    # `eval` and `affine_eval` are functions that can be called
    def __init__(self, style):
    
        if style not in ['classify-only', 'classify-and-distance']:
            raise ValueError("unrecognized style")

        self.style = style

    def __call__(self, params, x):
        raise RuntimeError("ImplicitFunction does not implement a __call__() operator. Subclasses must provide an implementation if is to be used.")

    def classify_box(self, params, box_lower, box_upper, offset=0.):
        '''
        Determine the sign of the function within a box (reports one of SIGN_UNKNOWN, etc)
        '''

        # delegate to the more general version
        center = 0.5 * (box_lower + box_upper)
        pos_vec = box_upper - center
        vecs = jnp.diag(pos_vec)
        return self.classify_general_box(params, center, vecs, offset=offset)

    # General version for non-axis-aligned boxes
    def classify_general_box(self, params, box_center, box_vecs, offset=0.):
        '''
        Determine the sign of the function within a general box (reports one of SIGN_UNKNOWN, etc)
        '''

        raise RuntimeError("ImplicitFunction does not implement classify_general_box(). Subclasses must provide an implementation if is to be used.")


    def min_distance_to_zero(self, params, box_center, box_axis_vec):
        '''
        Computes a lower bound on the distance to 0 from the center the box defined by `box_center` and `box_vecs`. The result is signed, a positive value means the function at the center point is positive, and likewise for negative.

        The query is evaluated on the axis-aligned range defined by the nonnegative values `box_vecs`. The min(box_vecs) is the largest-magnitude value which can ever be returned.

        If `box_vecs`, is `None`, it will be treated as the infinite domain. (Though some subclasses may not support this).
        '''

        raise RuntimeError("ImplicitFunction does not implement min_distance_to_zero(). Subclasses must provide an implementation if is to be used.")
   

    def min_distance_to_zero_in_direction(self, params, source_point, bound_vec, source_range=None, return_source_value=False):
        '''
        Computes a lower bound on the distance to 0 from `source_point` point in the direction `bound_vec`. The query is evaluated on the range `[source_point, source_point+bound_vec]`, and the magnitude of `bound_vec` is the largest-magnitude value which can be returned. 

        Optionally, `source_range` is a `(v,d)` array of vectors defining a general box in space over which to evaluate the query. These vectors must be orthogonal to `bound_vec`. The result is then a minimum over all direction vectors in that prisim. 

        Many methods incidentally compute the value of the function at the source as a part of evaluating this routine. If `return_source_value=True` the return will be a tuple `value, distance` giving the value as well.
        '''


        raise RuntimeError("ImplicitFunction does not implement min_distance_to_zero_in_direction(). Subclasses must provide an implementation if is to be used.")

   
