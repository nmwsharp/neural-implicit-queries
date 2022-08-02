import jax
import jax.numpy as jnp

import numpy as np

def norm(x):
    return jnp.linalg.norm(x, axis=-1)

def norm2(x):
    return jnp.inner(x,x)

def normalize(x):
    return x / norm(x)

def orthogonal_dir(x, remove_dir):
    # take a vector x, remove any component in the direction of vector remove_dir, and return unit x
    remove_dir = normalize(remove_dir)
    x = x - jnp.dot(x, remove_dir) * remove_dir
    return normalize(x)

def dot(x,y):
    return jnp.sum(x*y, axis=-1)

def normalize_positions(pos, method='bbox'):
    # center and unit-scale positions in to the [-1,1] cube

    if method == 'mean':
        # center using the average point position
        pos = pos - jnp.mean(pos, axis=-2, keepdims=True)
    elif method == 'bbox': 
        # center via the middle of the axis-aligned bounding box
        bbox_min = jnp.min(pos, axis=-2)
        bbox_max = jnp.max(pos, axis=-2)
        center = (bbox_max + bbox_min) / 2.
        pos -= center[None,:]
    else:
        raise ValueError("unrecognized method")

    scale = jnp.max(norm(pos), axis=-1, keepdims=True)[:,None]
    pos = pos / scale
    return pos

def sample_mesh_sdf(V, F, n_sample, surface_frac=0.5, surface_perturb_sigma=0.01, ambient_expand=1.25):
    import igl
    '''
    NOTE: Assumes input is scaled to lie in [-1,1] cube
    NOTE: RNG is handled internally, in part by an external library (libigl). Has none of the usual JAX RNG properties, may or may not yield same results, etc.
    '''

    n_surface = int(n_sample * surface_frac)
    n_ambient = n_sample - n_surface

    # Compute a bounding box for the mesh
    bbox_min = np.array([-1,-1,-1])
    bbox_max = np.array([1,1,1])
    center = 0.5*(bbox_max + bbox_min)

    # Sample ambient points
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # Q_ambient = jax.random.normal(subkey, (n_ambient, 3)) * ambient_sigma
    Q_ambient = jax.random.uniform(subkey, (n_ambient, 3), minval=-ambient_expand, maxval=ambient_expand)

    # Sample surface points
    sample_b, sample_f = igl.random_points_on_mesh(n_surface, np.array(V), np.array(F))
    face_verts = V[F[sample_f], :]
    raw_samples = np.sum(sample_b[...,np.newaxis] * face_verts, axis=1)
    raw_samples = jnp.array(raw_samples)

    # add noise to surface points
    key, subkey = jax.random.split(key)
    offsets = jax.random.normal(subkey, (n_surface, 3)) * surface_perturb_sigma
    Q_surface = raw_samples + offsets

    # Combine and shuffle
    Q = np.vstack((Q_ambient, Q_surface))
    key, subkey = jax.random.split(key)
    Q = jax.random.permutation(subkey, Q, axis=0)

    # Get SDF value via distance & winding number
    sdf_vals, _, closest = igl.signed_distance(np.array(Q), np.array(V), np.array(F))
    sdf_vals = jnp.array(sdf_vals)

    return Q, sdf_vals


def sample_mesh_importance(V, F, n_sample, n_sample_full_mult=10., beta=20., ambient_range=1.25):
    import igl

    V = np.array(V)
    F = np.array(F)
    n_sample_full = int(n_sample * n_sample_full_mult)

    # Sample ambient points
    Q_ambient = np.random.uniform(size=(n_sample_full, 3), low=-ambient_range, high=ambient_range)

    # Assign weights 
    dist_sq, _, _ = igl.point_mesh_squared_distance(Q_ambient, np.array(V), np.array(F))
    weight = np.exp(-beta * np.sqrt(dist_sq))
    weight = weight / np.sum(weight)

    # Sample
    samp_inds = np.random.choice(n_sample_full, size=n_sample, p=weight)
    Q = Q_ambient[samp_inds,:]

    # Get SDF value via distance & winding number
    sdf_vals, _, closest = igl.signed_distance(Q, V, F)
    sdf_vals = jnp.array(sdf_vals)
    Q = jnp.array(Q)

    return Q, sdf_vals
