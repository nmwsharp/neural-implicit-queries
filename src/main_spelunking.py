import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
from functools import partial

import jax
import jax.numpy as jnp

import argparse
import matplotlib
import matplotlib.pyplot as plt
import imageio
from skimage import measure


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
from kd_tree import *
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
import implicit_mlp_utils, extract_cell
import affine_layers
import slope_interval_layers

# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def save_render_current_view(args, implicit_func, params, cast_frustum, opts, matcaps, surf_color):

    root = ps.get_camera_world_position()
    look, up, left = ps.get_camera_frame()
    fov_deg = ps.get_field_of_view()
    res = args.res // opts['res_scale']

    surf_color = tuple(surf_color)

    img, depth, count, _, eval_sum, raycast_time = render.render_image(implicit_func, params, root, look, up, left, res, fov_deg, cast_frustum, opts, shading='matcap_color', matcaps=matcaps, shading_color_tuple=(surf_color,))

    # flip Y
    img = img[::-1,:,:]

    # append an alpha channel
    alpha_channel = (jnp.min(img,axis=-1) < 1.) * 1.
    # alpha_channel = jnp.ones_like(img[:,:,0])
    img_alpha = jnp.concatenate((img, alpha_channel[:,:,None]), axis=-1)
    img_alpha = jnp.clip(img_alpha, a_min=0., a_max=1.)
    img_alpha = np.array(img_alpha)
    print(f"Saving image to {args.image_write_path}")
    imageio.imwrite(args.image_write_path, img_alpha)


def do_sample_surface(opts, implicit_func, params, n_samples, sample_width, n_node_thresh, do_viz_tree, do_uniform_sample):
    data_bound = opts['data_bound']
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))

    rngkey = jax.random.PRNGKey(0)

    print(f"do_sample_surface n_node_thresh {n_node_thresh}")

    with Timer("sample points"):
        sample_points = sample_surface(implicit_func, params, lower, upper, n_samples, sample_width, rngkey, n_node_thresh=n_node_thresh)
        sample_points.block_until_ready()

    ps.register_point_cloud("sampled points", np.array(sample_points))


    # Build the tree all over again so we can visualize it
    if do_viz_tree:
        out_dict = construct_uniform_unknown_levelset_tree(implicit_func, params, lower, upper, n_node_thresh, offset=sample_width)
        node_valid = out_dict['unknown_node_valid']
        node_lower = out_dict['unknown_node_lower']
        node_upper = out_dict['unknown_node_upper']
        node_lower = node_lower[node_valid,:]
        node_upper = node_upper[node_valid,:]
        verts, inds = generate_tree_viz_nodes_simple(node_lower, node_upper, shrink_factor=0.05)
        ps_vol = ps.register_volume_mesh("tree nodes", np.array(verts), hexes=np.array(inds))

    # If requested, also do uniform sampling
    if do_uniform_sample:

        with Timer("sample points uniform"):
            sample_points = sample_surface_uniform(implicit_func, params, lower, upper, n_samples, sample_width, rngkey)
            sample_points.block_until_ready()

        ps.register_point_cloud("uniform sampled points", np.array(sample_points))



def do_hierarchical_mc(opts, implicit_func, params, n_mc_depth, do_viz_tree, compute_dense_cost):


    data_bound = opts['data_bound']
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))


    print(f"do_hierarchical_mc {n_mc_depth}")
    

    with Timer("extract mesh"):
        tri_pos = hierarchical_marching_cubes(implicit_func, params, lower, upper, n_mc_depth, n_subcell_depth=3)
        tri_pos.block_until_ready()

    tri_inds = jnp.reshape(jnp.arange(3*tri_pos.shape[0]), (-1,3))
    tri_pos = jnp.reshape(tri_pos, (-1,3))
    ps.register_surface_mesh("extracted mesh", np.array(tri_pos), np.array(tri_inds))

    # Build the tree all over again so we can visualize it
    if do_viz_tree:
        n_mc_subcell=3
        out_dict = construct_uniform_unknown_levelset_tree(implicit_func, params, lower, upper, split_depth=3*(n_mc_depth-n_mc_subcell), with_interior_nodes=True, with_exterior_nodes=True)

        node_valid = out_dict['unknown_node_valid']
        node_lower = out_dict['unknown_node_lower']
        node_upper = out_dict['unknown_node_upper']
        node_lower = node_lower[node_valid,:]
        node_upper = node_upper[node_valid,:]
        verts, inds = generate_tree_viz_nodes_simple(node_lower, node_upper, shrink_factor=0.05)
        ps_vol = ps.register_volume_mesh("unknown tree nodes", np.array(verts), hexes=np.array(inds))

        node_valid = out_dict['interior_node_valid']
        node_lower = out_dict['interior_node_lower']
        node_upper = out_dict['interior_node_upper']
        node_lower = node_lower[node_valid,:]
        node_upper = node_upper[node_valid,:]
        if node_lower.shape[0] > 0:
            verts, inds = generate_tree_viz_nodes_simple(node_lower, node_upper, shrink_factor=0.05)
            ps_vol = ps.register_volume_mesh("interior tree nodes", np.array(verts), hexes=np.array(inds))
        
        node_valid = out_dict['exterior_node_valid']
        node_lower = out_dict['exterior_node_lower']
        node_upper = out_dict['exterior_node_upper']
        node_lower = node_lower[node_valid,:]
        node_upper = node_upper[node_valid,:]
        if node_lower.shape[0] > 0:
            verts, inds = generate_tree_viz_nodes_simple(node_lower, node_upper, shrink_factor=0.05)
            ps_vol = ps.register_volume_mesh("exterior tree nodes", np.array(verts), hexes=np.array(inds))

def do_closest_point(opts, func, params, n_closest_point):

    data_bound = float(opts['data_bound'])
    eps = float(opts['hit_eps'])
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound,   data_bound,  data_bound))

    print(f"do_closest_point {n_closest_point}")
   
    # generate some query points
    rngkey = jax.random.PRNGKey(n_closest_point)
    rngkey, subkey = jax.random.split(rngkey)
    query_points = jax.random.uniform(subkey, (n_closest_point,3), minval=lower, maxval=upper)

    with Timer("closest point"):
        query_dist, query_min_loc = closest_point(func, params, lower, upper, query_points, eps=eps)
        query_dist.block_until_ready()

    # visualize only the outside ones
    is_outside = jax.vmap(partial(func,params))(query_points) > 0
    query_points = query_points[is_outside,:]
    query_dist = query_dist[is_outside]
    query_min_loc = query_min_loc[is_outside,:]

    viz_line_nodes = jnp.reshape(jnp.stack((query_points, query_min_loc), axis=1), (-1,3))
    viz_line_edges = jnp.reshape(jnp.arange(2*query_points.shape[0]), (-1,2))
    ps.register_point_cloud("closest point query", np.array(query_points))
    ps.register_point_cloud("closest point result", np.array(query_min_loc))
    ps.register_curve_network("closest point line", np.array(viz_line_nodes), np.array(viz_line_edges))


def compute_bulk(args, implicit_func, params, opts):
    
    data_bound = float(opts['data_bound'])
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))
        
    rngkey = jax.random.PRNGKey(0)

    with Timer("bulk properties"):
        mass, centroid = bulk_properties(implicit_func, params, lower, upper, rngkey)
        mass.block_until_ready()

    print(f"Bulk properties:")
    print(f"  Mass: {mass}")
    print(f"  Centroid: {centroid}")

    ps.register_point_cloud("centroid", np.array([centroid]))

def main():

    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    
    parser.add_argument("--res", type=int, default=1024)
    
    parser.add_argument("--image_write_path", type=str, default="render_out.png")

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


    # GUI Parameters
    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    cast_frustum = False
    mode = 'affine_fixed'
    modes = ['sdf', 'interval', 'affine_fixed', 'affine_truncate', 'affine_append', 'affine_all', 'slope_interval']
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 8
    affine_opts['affine_n_append'] = 4
    affine_opts['sdf_lipschitz'] = 1.
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = 'absolute'
    n_sample_pts = 100000
    sample_width = 0.01
    n_node_thresh = 4096
    do_uniform_sample = False
    do_viz_tree = False
    n_mc_depth = 8
    compute_dense_cost = False
    n_closest_point = 16
    shade_style = 'matcap_color'
    surf_color = (0.157,0.613,1.000)

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    # load the matcaps
    matcaps = render.load_matcap(os.path.join(ROOT_DIR, "assets", "matcaps", "wax_{}.png"))

    def callback():

        nonlocal implicit_func, params, mode, modes, cast_frustum, debug_log_compiles, debug_disable_jit, debug_debug_nans, shade_style, surf_color, n_sample_pts, sample_width, n_node_thresh, do_uniform_sample, do_viz_tree, n_mc_depth, compute_dense_cost, n_closest_point
            
    
        ## Options for general affine evaluation
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Eval options"):
            psim.PushItemWidth(100)
    
            old_mode = mode
            changed, mode = utils.combo_string_picker("Method", mode, modes)
            if mode != old_mode:
                implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

            if mode == 'affine_truncate':
                # truncate options

                changed, affine_opts['affine_n_truncate'] = psim.InputInt("affine_n_truncate", affine_opts['affine_n_truncate'])
                if changed: 
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
            
                changed, affine_opts['affine_truncate_policy'] = utils.combo_string_picker("Method", affine_opts['affine_truncate_policy'], truncate_policies)
                if changed:
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
            
            if mode == 'affine_append':
                # truncate options

                changed, affine_opts['affine_n_append'] = psim.InputInt("affine_n_append", affine_opts['affine_n_append'])
                if changed: 
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
            
            if mode == 'sdf':

                changed, affine_opts['sdf_lipschitz'] = psim.InputFloat("SDF Lipschitz", affine_opts['sdf_lipschitz'])
                if changed:
                    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)


            psim.PopItemWidth()
            psim.TreePop()


        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Raycast"):
            psim.PushItemWidth(100)
        
            if psim.Button("Save Render"):
                save_render_current_view(args, implicit_func, params, cast_frustum, opts, matcaps, surf_color)
            

            _, cast_frustum = psim.Checkbox("cast frustum", cast_frustum)
            _, opts['hit_eps'] = psim.InputFloat("delta", opts['hit_eps'])
            _, opts['max_dist'] = psim.InputFloat("max_dist", opts['max_dist'])

            if cast_frustum:
                _, opts['n_side_init'] = psim.InputInt("n_side_init", opts['n_side_init'])

            psim.PopItemWidth()
            psim.TreePop()


        # psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Sample Surface "):
            psim.PushItemWidth(100)

            if psim.Button("Sample"):
                do_sample_surface(opts, implicit_func, params, n_sample_pts, sample_width, n_node_thresh, do_viz_tree, do_uniform_sample)

            _, n_sample_pts = psim.InputInt("n_sample_pts", n_sample_pts)

            psim.SameLine()
            _, sample_width = psim.InputFloat("sample_width", sample_width)
            _, n_node_thresh = psim.InputInt("n_node_thresh", n_node_thresh)
            _, do_viz_tree = psim.Checkbox("viz tree", do_viz_tree)
            psim.SameLine()
            _, do_uniform_sample = psim.Checkbox("also uniform sample", do_uniform_sample)

            
            psim.PopItemWidth()
            psim.TreePop()


        if psim.TreeNode("Extract mesh"):
            psim.PushItemWidth(100)

            if psim.Button("Extract"):
                do_hierarchical_mc(opts, implicit_func, params, n_mc_depth, do_viz_tree, compute_dense_cost)

            psim.SameLine()
            _, n_mc_depth = psim.InputInt("n_mc_depth", n_mc_depth)
            _, do_viz_tree = psim.Checkbox("viz tree", do_viz_tree)
            psim.SameLine()
            _, compute_dense_cost = psim.Checkbox("compute dense cost", compute_dense_cost)

            
            psim.PopItemWidth()
            psim.TreePop()
       

        if psim.TreeNode("Closest point"):
            psim.PushItemWidth(100)

            if psim.Button("Find closest pionts"):
                do_closest_point(opts, implicit_func, params, n_closest_point)

            _, n_closest_point= psim.InputInt("n_closest_point", n_closest_point)
            
            psim.PopItemWidth()
            psim.TreePop()

        ## Bulk
        if psim.TreeNode("Bulk Properties"):
            psim.PushItemWidth(100)
        
            if psim.Button("Compute bulk"):
                compute_bulk(args, implicit_func, params, opts)

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
               
    ps.set_use_prefs_file(False)
    ps.init()



    # Visualize the data via quick coarse marching cubes, so we have something to look at

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
    ps.register_surface_mesh("coarse shape preview", verts, faces) 
   
    print("REMEMBER: All routines will be slow on the first invocation due to JAX kernel compilation. Subsequent calls will be fast.")

    # Hand off control to the main callback
    ps.show(1)
    ps.set_user_callback(callback)
    ps.show()


if __name__ == '__main__':
    main()
