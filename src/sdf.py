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

class WeakSDFImplicitFunction(implicit_function.ImplicitFunction):

    def __init__(self, sdf_func, lipschitz_bound=1.):
        super().__init__("classify-only")
        self.sdf_func = sdf_func
        self.lipschitz_bound = lipschitz_bound

    def __call__(self, params, x):
        return self.sdf_func(params, x)

    # the parent class automatically delegates to this
    # def classify_box(self, params, box_lower, box_upper):
        # pass
        
    def classify_general_box(self, params, box_center, box_vecs, offset=0.):

        # compute the radius of the box
        rad = jnp.sqrt(jnp.sum(jnp.square(jnp.linalg.norm(box_vecs, axis=-1)), axis=0))

        d = box_center.shape[-1]
        v = box_vecs.shape[-2]
        assert box_center.shape == (d,), "bad box_vecs shape"
        assert box_vecs.shape == (v,d), "bad box_vecs shape"

        # evaluate the function
        val = self.sdf_func(params, box_center)
        can_change = jnp.abs(val) - rad * self.lipschitz_bound < 0.

        # determine the type of the region
        output_type = SIGN_UNKNOWN
        output_type = jnp.where(jnp.logical_and(~can_change, val >  offset), SIGN_POSITIVE, output_type)
        output_type = jnp.where(jnp.logical_and(~can_change, val < -offset), SIGN_NEGATIVE, output_type)

        return output_type
