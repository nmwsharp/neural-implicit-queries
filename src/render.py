import os
import sys

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
from functools import partial
import imageio

import geometry
import queries
from utils import *
import affine

# theta_x/y should be 
def camera_ray(look_dir, up_dir, left_dir, fov_deg_x, fov_deg_y, theta_x, theta_y):
    ray_image_plane_pos = look_dir \
                          + left_dir * (theta_x * jnp.tan(jnp.deg2rad(fov_deg_x)/2)) \
                          + up_dir *    (theta_y * jnp.tan(jnp.deg2rad(fov_deg_y)/2))

    ray_dir = geometry.normalize(ray_image_plane_pos)

    return ray_dir

@partial(jax.jit, static_argnames=("res"))
def generate_camera_rays(eye_pos, look_dir, up_dir, res=1024, fov_deg=30.):

    D = res     # image dimension
    R = res*res # number of rays

    ## Generate rays according to a pinhole camera

    # Image coords on [-1,1] for each output pixel
    cam_ax_x = jnp.linspace(-1., 1., res)
    cam_ax_y = jnp.linspace(-1., 1., res)
    cam_x, cam_y = jnp.meshgrid(cam_ax_x, cam_ax_y)
    cam_x = cam_x.flatten() # [R]
    cam_y = cam_y.flatten() # [R]

    # Orthornormal camera frame
    up_dir = up_dir - jnp.dot(look_dir, up_dir) * look_dir
    up_dir = geometry.normalize(up_dir)
    left_dir = jnp.cross(look_dir, up_dir)


    # ray_roots, ray_dirs = jax.jit(jax.vmap(camera_ray))(cam_x, cam_y)
    ray_dirs = jax.vmap(partial(camera_ray, look_dir, up_dir, left_dir, fov_deg, fov_deg))(cam_x, cam_y)
    ray_roots = jnp.tile(eye_pos, (ray_dirs.shape[0],1))
    return ray_roots, ray_dirs
    

@partial(jax.jit, static_argnames=("funcs_tuple", "method"))
def outward_normal(funcs_tuple, params_tuple, hit_pos, hit_id, eps, method='finite_differences'):

    grad_out = jnp.zeros(3)
    i_func = 1
    for func, params in zip(funcs_tuple, params_tuple):
        f = partial(func, params)

        if method == 'autodiff':
            grad_f = jax.jacfwd(f)
            grad = grad_f(hit_pos)

        elif method == 'finite_differences':
            # 'tetrahedron' central differences approximation
            # see e.g. https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
            offsets = jnp.array((
                (+eps, -eps, -eps),
                (-eps, -eps, +eps),
                (-eps, +eps, -eps),
                (+eps, +eps, +eps),
                ))
            x_pts = hit_pos[None,:] + offsets
            samples = jax.vmap(f)(x_pts)
            grad = jnp.sum(offsets * samples[:,None], axis=0)

        else:
            raise ValueError("unrecognized method")

        grad = geometry.normalize(grad)
        grad_out = jnp.where(hit_id == i_func, grad, grad_out)
        i_func += 1
        
    return grad_out

@partial(jax.jit, static_argnames=("funcs_tuple", "method"))
def outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, eps, method='finite_differences'):
    this_normal_one = lambda p, id : outward_normal(funcs_tuple, params_tuple, p, id, eps, method=method)
    return jax.vmap(this_normal_one)(hit_pos, hit_ids)


# @partial(jax.jit, static_argnames=("func","res"))
def render_image(funcs_tuple, params_tuple, eye_pos, look_dir, up_dir, left_dir, res, fov_deg, frustum, opts, shading="normal", shading_color_tuple=((0.157,0.613,1.000)), matcaps=None, tonemap=False, shading_color_func=None):

    # make sure inputs are tuples not lists (can't has lists)
    if isinstance(funcs_tuple, list): funcs_tuple = tuple(funcs_tuple)
    if isinstance(params_tuple, list): params_tuple = tuple(params_tuple)
    if isinstance(shading_color_tuple, list): shading_color_tuple = tuple(shading_color_tuple)

    # wrap in tuples if single was passed
    if not isinstance(funcs_tuple, tuple):
        funcs_tuple = (funcs_tuple,)
    if not isinstance(params_tuple, tuple):
        params_tuple = (params_tuple,)
    if not isinstance(shading_color_tuple[0], tuple):
        shading_color_tuple = (shading_color_tuple,)

    L = len(funcs_tuple)
    if (len(params_tuple) != L) or (len(shading_color_tuple) != L):
        raise ValueError("render_image tuple arguments should all be same length")

    ray_roots, ray_dirs = generate_camera_rays(eye_pos, look_dir, up_dir, res=res, fov_deg=fov_deg)
    if frustum:
        # == Frustum raycasting
            
        cam_params = eye_pos, look_dir, up_dir, left_dir, fov_deg, fov_deg, res, res 

        with Timer("frustum raycast"):
            t_raycast, hit_ids, counts, n_eval = queries.cast_rays_frustum(funcs_tuple, params_tuple, cam_params, opts)
            t_raycast.block_until_ready()

        # TODO transposes here due to image layout conventions. can we get rid of them?
        t_raycast = t_raycast.transpose().flatten()
        hit_ids = hit_ids.transpose().flatten()
        counts = counts.transpose().flatten()

    else:
        # == Standard raycasting
        with Timer("raycast"):
            t_raycast, hit_ids, counts, n_eval = queries.cast_rays(funcs_tuple, params_tuple, ray_roots, ray_dirs, opts)
            t_raycast.block_until_ready()
   

    hit_pos = ray_roots + t_raycast[:,np.newaxis] * ray_dirs
    hit_normals = outward_normals(funcs_tuple, params_tuple, hit_pos, hit_ids, opts['hit_eps'])
    hit_color = shade_image(shading, ray_dirs, hit_pos, hit_normals, hit_ids, up_dir, matcaps, shading_color_tuple, shading_color_func=shading_color_func)

    img = jnp.where(hit_ids[:,np.newaxis], hit_color, jnp.ones((res*res, 3)))

    if tonemap:
        # We intentionally tonemap before compositing in the shadow. Otherwise the white level clips the shadow and gives it a hard edge.
        img = tonemap_image(img)

    img = img.reshape(res,res,3)
    depth = t_raycast.reshape(res,res)
    counts = counts.reshape(res,res)
    hit_ids = hit_ids.reshape(res,res)

    return img, depth, counts, hit_ids, n_eval, -1

def tonemap_image(img, gamma=2.2, white_level=.75, exposure=1.):
    img = img * exposure
    num = img * (1.0 + (img / (white_level * white_level)))
    den = (1.0 + img)
    img = num / den;
    img = jnp.power(img, 1.0/gamma)
    return img

@partial(jax.jit, static_argnames=("shading", "shading_color_func"))
def shade_image(shading, ray_dirs, hit_pos, hit_normals, hit_ids, up_dir, matcaps, shading_color_tuple, shading_color_func=None):

    # Simple shading
    if shading == "normal":
        hit_color = (hit_normals + 1.) / 2. # map normals to [0,1]

    elif shading == "matcap_color":

        # compute matcap coordinates
        ray_up = jax.vmap(partial(geometry.orthogonal_dir,up_dir))(ray_dirs)
        ray_left = jax.vmap(jnp.cross)(ray_dirs, ray_up)
        matcap_u = jax.vmap(jnp.dot)(-ray_left, hit_normals)
        matcap_v = jax.vmap(jnp.dot)(ray_up, hit_normals)

        # pull inward slightly to avoid indexing off the matcap image
        matcap_u *= .98
        matcap_v *= .98

        # remap to image indices 
        matcap_x = (matcap_u + 1.) / 2. * matcaps[0].shape[0]
        matcap_y = (-matcap_v + 1.) / 2. * matcaps[0].shape[1]
        matcap_coords = jnp.stack((matcap_x, matcap_y), axis=0)

        def sample_matcap(matcap, coords):
            m = lambda X : jax.scipy.ndimage.map_coordinates(X, coords, order=1, mode='nearest')
            return jax.vmap(m, in_axes=-1, out_axes=-1)(matcap)

        # fetch values
        mat_r = sample_matcap(matcaps[0], matcap_coords)
        mat_g = sample_matcap(matcaps[1], matcap_coords)
        mat_b = sample_matcap(matcaps[2], matcap_coords)
        mat_k = sample_matcap(matcaps[3], matcap_coords)

        # find the appropriate shading color
        def get_shade_color(hit_pos, hit_id):
            shading_color = jnp.ones(3)

            if shading_color_func is None:
                # use the tuple of constant colors
                i_func = 1
                for c in shading_color_tuple:
                    shading_color = jnp.where(hit_id == i_func, jnp.array(c), shading_color)
                    i_func += 1
            else:
                # look up varying color
                shading_color = shading_color_func(hit_pos)

            return shading_color
        shading_color = jax.vmap(get_shade_color)(hit_pos, hit_ids)

        c_r, c_g, c_b = shading_color[:,0], shading_color[:,1], shading_color[:,2]
        c_k = 1. - (c_r + c_b + c_g)

        c_r = c_r[:,None]
        c_g = c_g[:,None]
        c_b = c_b[:,None]
        c_k = c_k[:,None]

        hit_color = c_r*mat_r + c_b*mat_b + c_g*mat_g + c_k*mat_k

    else:
        raise RuntimeError("Unrecognized shading parameter")

    return hit_color



# create camera parameters looking in a direction
def look_at(eye_pos, target=None, up_dir='y'):

    if target == None:
        target = jnp.array((0., 0., 0.,))
    if up_dir == 'y':
        up_dir = jnp.array((0., 1., 0.,))
    elif up_dir == 'z':
        up_dir = jnp.array((0., 0., 1.,))

    look_dir = geometry.normalize(target - eye_pos)
    up_dir = geometry.orthogonal_dir(up_dir, look_dir)
    left_dir = jnp.cross(look_dir, up_dir)

    return look_dir, up_dir, left_dir


def load_matcap(fname_pattern):

    imgs = []
    for c in ['r', 'g', 'b', 'k']:
        im = imageio.imread(fname_pattern.format(c))
        im = jnp.array(im) / 256.
        imgs.append(im)

    return tuple(imgs)
