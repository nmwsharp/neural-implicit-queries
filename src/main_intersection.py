import sys, os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from skimage import measure

import argparse

import polyscope as ps
import polyscope.imgui as psim


# Imports from this project
import render, geometry, queries
from geometry import *
from utils import *
import affine
import slope_interval
import sdf
import mlp
import kd_tree 
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE

import affine_layers
import slope_interval_layers
import implicit_mlp_utils

# Config
# from jax.config import config 

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def main():

    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("inputA", type=str)
    parser.add_argument("inputB", type=str)
    
    parser.add_argument("--res", type=int, default=1024)
    
    parser.add_argument("--mode", type=str, default='affine_all')
    
    parser.add_argument("--scaleA", type=float, default=1.)
    parser.add_argument("--scaleB", type=float, default=1.)

    parser.add_argument("--log-compiles", action='store_true')
    parser.add_argument("--disable-jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable-double-precision", action='store_true')

    # Parse arguments
    args = parser.parse_args()

    ## Small options
    debug_log_compiles = False
    debug_disable_jit = False
    debug_debug_nans = False
    if args.log_compiles:
        jax.config.update("jax_log_compiles", 1)
        debug_log_compiles = True
    if args.disable_jit:
        jax.config.update('jax_disable_jit', True)
        debug_disable_jit = True
    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
        debug_debug_nans = True
    if args.enable_double_precision:
        jax.config.update("jax_enable_x64", True)

    ps.set_use_prefs_file(False)
    ps.init()

    # GUI Parameters
    continuously_render = False
    fancy_render = False
    continuously_intersect = False
    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['intersection_eps'] = 1e-3
    cast_frustum = False
    shade_style = 'matcap_color'
    surf_colorA = (0.157,0.613,1.000)
    surf_colorB = (0.215,0.865,0.046)
    

    # Load the shapes
    print("Loading shapes")
    implicit_funcA, paramsA = implicit_mlp_utils.generate_implicit_from_file(args.inputA, mode=args.mode, affine_n_truncate=64, affine_truncate_policy='absolute')
    paramsA = mlp.prepend_op(paramsA, mlp.spatial_transformation()) 

    implicit_funcB, paramsB = implicit_mlp_utils.generate_implicit_from_file(args.inputB, mode=args.mode, affine_n_truncate=64, affine_truncate_policy='absolute')
    paramsB = mlp.prepend_op(paramsB, mlp.spatial_transformation()) 


    # Register volume quantities in Polyscope for the shapes
    def register_volume(name, implicit_func, params, scale=1.):  

        # Construct the regular grid
        grid_res = 128
        ax_coords = jnp.linspace(-1., 1., grid_res)
        grid_x, grid_y, grid_z = jnp.meshgrid(ax_coords, ax_coords, ax_coords, indexing='ij')
        grid = jnp.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)
        delta = (grid[1,2] - grid[0,2]).item()
        sdf_vals = jax.vmap(partial(implicit_func, params))(grid)
        sdf_vals = sdf_vals.reshape(grid_res, grid_res, grid_res)
        bbox_min = grid[0,:]
        verts, faces, normals, values = measure.marching_cubes(np.array(sdf_vals), level=0., spacing=(delta, delta, delta))
        verts = verts + bbox_min[None,:]
        ps_surf = ps.register_surface_mesh(name, verts, faces) 
        return ps_surf

    print("Registering grids")
    ps_vol_A = register_volume("shape A coarse preview", implicit_funcA, paramsA, args.scaleA)
    ps_vol_B = register_volume("shape B coarse preview", implicit_funcB, paramsB, args.scaleB)
    
    print("Loading matcaps")
    matcaps = render.load_matcap(os.path.join(ROOT_DIR, "assets", "matcaps", "wax_{}.png"))

    print("Done")
    def find_intersection():

        func_tuple = (implicit_funcA, implicit_funcB)
        params_tuple = (paramsA, paramsB)
        data_bound = opts['data_bound']
        lower = jnp.array((-data_bound, -data_bound, -data_bound))
        upper = jnp.array((data_bound, data_bound, data_bound))
        eps = opts['intersection_eps']

        with Timer("intersection"):
            found_int, found_int_A, found_int_B, found_int_loc = kd_tree.find_any_intersection(func_tuple, params_tuple, lower, upper, eps)

        if found_int:
            pos = np.array(found_int_loc)[None,:]
            ps_int_cloud = ps.register_point_cloud("intersection location", pos, enabled=True, radius=0.01, color=(1., 0., 0.))
        else:
            ps.remove_point_cloud("intersection location", error_if_absent=False)


    def viz_intersection_tree():

        func_tuple = (implicit_funcA, implicit_funcB)
        params_tuple = (paramsA, paramsB)
        data_bound = opts['data_bound']
        lower = jnp.array((-data_bound, -data_bound, -data_bound))
        upper = jnp.array((data_bound, data_bound, data_bound))
        eps = opts['intersection_eps']

        found_int, found_int_A, found_int_B, found_int_loc, nodes_lower, nodes_upper, nodes_type = kd_tree.find_any_intersection(func_tuple, params_tuple, lower, upper, eps, viz_nodes=True)

        
        verts, inds = kd_tree.generate_tree_viz_nodes_simple(nodes_lower, nodes_upper)
        
        ps_vol_nodes = ps.register_volume_mesh("search tree nodes", np.array(verts), hexes=np.array(inds))
        ps_vol_nodes.add_scalar_quantity("type", np.array(nodes_type), defined_on='cells')
        ps_vol_nodes.set_enabled(True)


    def callback():

        nonlocal implicit_funcA, paramsA, implicit_funcB, paramsB, continuously_render, fancy_render, continuously_intersect, cast_frustum, debug_log_compiles, debug_disable_jit, debug_debug_nans, shade_style, surf_colorA, surf_colorB

        
        # === Update transforms from Polyscope
        def update_transform(ps_vol, params, scale=1.):
            T = ps_vol.get_transform()
            R = T[:3,:3]

            # TODO this absurdity makes it the transform behave as expected.
            # I think there just miiiiight be a bug in the transforms Polyscope is returning
            R_inv = jnp.linalg.inv(R)
            t = R_inv @ R_inv @ T[:3,3]

            params["0000.spatial_transformation.R"] = R_inv * scale
            params["0000.spatial_transformation.t"] = t
        update_transform(ps_vol_A, paramsA, args.scaleA)
        update_transform(ps_vol_B, paramsB, args.scaleB)

        # === Build the UI
        
        ## Intersection & options
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Intersection"):
            psim.PushItemWidth(100)
        
            if psim.Button("Check for intersection"):
                find_intersection()
            psim.SameLine()

            _, continuously_intersect = psim.Checkbox("Continuously intersect", continuously_intersect)
            if continuously_intersect:
                find_intersection()
            

            _, opts['intersection_eps'] = psim.InputFloat("intersection_delta", opts['intersection_eps'])
            

            if psim.Button("Viz intersection tree"):
                viz_intersection_tree()

            psim.PopItemWidth()
            psim.TreePop()
        
        

        if psim.TreeNode("Debug"):
            psim.PushItemWidth(100)

            changed, debug_log_compiles = psim.Checkbox("debug_log_compiles", debug_log_compiles)
            if changed:
                jax.config.update("jax_log_compiles", 1 if debug_log_compiles else 0)

            changed, debug_disable_jit = psim.Checkbox("debug_disable_jit", debug_disable_jit)
            if changed:
                jax.config.update('jax_disable_jit', debug_disable_jit)
            
            changed, debug_debug_nans = psim.Checkbox("debug_debug_nans", debug_debug_nans)
            if changed:
                jax.config.update("jax_debug_nans", debug_debug_nans)

            
            psim.PopItemWidth()
            psim.TreePop()


    # Hand off control to the main callback
    ps.set_user_callback(callback)
    ps.show()



if __name__ == '__main__':
    main()
