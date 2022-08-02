import igl

import sys
from functools import partial
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

# Imports from this project
from utils import *
import mlp
import geometry
import render
import queries
import affine

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")

def main():

    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
   
    # network
    parser.add_argument("--activation", type=str, default='elu')
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--layer_width", type=int, default=32)
    parser.add_argument("--positional_encoding", action='store_true')
    parser.add_argument("--positional_count", type=int, default=10)
    parser.add_argument("--positional_pow_start", type=int, default=-3)

    # loss / data
    parser.add_argument("--fit_mode", type=str, default='sdf')
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=1000000)
    parser.add_argument("--sample_ambient_range", type=float, default=1.25)
    parser.add_argument("--sample_weight_beta", type=float, default=20.)
    
    # training
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr_decay_every", type=int, default=99999)
    parser.add_argument("--lr_decay_frac", type=float, default=.5)

    # jax options
    parser.add_argument("--log-compiles", action='store_true')
    parser.add_argument("--disable-jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable-double-precision", action='store_true')

    # Parse arguments
    args = parser.parse_args()

    # validate some inputs
    if args.activation not in ['relu', 'elu', 'cos']:
        raise ValueError("unrecognized activation")
    if args.fit_mode not in ['occupancy', 'sdf']:
        raise ValueError("unrecognized activation")
    if not args.output_file.endswith('.npz'):
        raise ValueError("output file should end with .npz")

    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())
   
    # Set jax things
    if args.log_compiles:
        jax.config.update("jax_log_compiles", 1)
    if args.disable_jit:
        jax.config.update('jax_disable_jit', True)
    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
    if args.enable_double_precision:
        jax.config.update("jax_enable_x64", True)
   
    # load the input
    print(f"Loading mesh {args.input_file}")
    V, F = igl.read_triangle_mesh(args.input_file)
    V = jnp.array(V)
    F = jnp.array(F)
    print(f"  ...done")

    # preprocess (center and scale)
    V = geometry.normalize_positions(V, method='bbox') 

    # sample training points
    print(f"Sampling {args.n_samples} training points...")
    # Uses a strategy which is basically the one Davies et al 
    # samp, samp_SDF = geometry.sample_mesh_sdf(V, F, args.n_samples, surface_frac=args.surface_frac, surface_perturb_sigma=args.surface_perturb_sigma, ambient_range=args.surface_ambient_range)
    samp, samp_SDF = geometry.sample_mesh_importance(V, F, args.n_samples, beta=args.sample_weight_beta, ambient_range=args.sample_ambient_range)

    if args.fit_mode == 'occupancy':
        samp_target = (samp_SDF > 0) * 1.0
        n_pos = jnp.sum(samp_target > 0)
        n_neg = samp_target.shape[0] - n_pos
        w_pos = n_neg / (n_pos + n_neg)
        w_neg = n_pos / (n_pos + n_neg)
        samp_weight = jnp.where(samp_target > 0, w_pos, w_neg)
    elif args.fit_mode in ['sdf', 'tanh']:
        samp_target = samp_SDF
        samp_weight = jnp.ones_like(samp_target)
    else: raise ValueError("bad arg")
    print(f"  ...done")

    # construct the network 
    print(f"Constructing {args.n_layers}x{args.layer_width} {args.activation} network...")
    if args.positional_encoding:
        spec_list = [mlp.pow2_frequency_encode(args.positional_count, start_pow=args.positional_pow_start, with_shift=True), mlp.sin()]
        layers = [6*args.positional_count] + [args.layer_width]*args.n_layers + [1]
        spec_list += mlp.quick_mlp_spec(layers, args.activation)
    else:
        layers = [3] + [args.layer_width]*args.n_layers + [1]
        spec_list = mlp.quick_mlp_spec(layers, args.activation)
    orig_params = mlp.build_spec(spec_list) 
    implicit_func = mlp.func_from_spec()


    # layer initialization
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    orig_params = mlp.initialize_params(orig_params, subkey)
    print(f"  ...done")

    # test eval to ensure the function isn't broken
    print(f"Network test evaluation...")
    implicit_func(orig_params, jnp.array((0.1, 0.2, 0.3)))
    print(f"...done")

    # Create an optimizer
    print(f"Creating optimizer...")
    def step_func(i_epoch): 
        out = args.lr * (args.lr_decay_frac ** (i_epoch // args.lr_decay_every))
        return out
    opt = optimizers.adam(step_func)

    opt_param_keys = mlp.opt_param_keys(orig_params)

    # Union our optimizable parameters with the non-optimizable ones
    def add_full_params(opt_params):
        all_params = opt_params
        
        for k in orig_params:
            if k not in all_params:
                all_params[k] = orig_params[k]
    
    # Union our optimizable parameters with the non-optimizable ones
    def filter_to_opt_params_only(all_params):
        for k in all_params:
            if k not in opt_param_keys:
                del all_params[k]
    
    # Construct the optimizer over the optimizable params
    opt_params_only = {}
    for k in mlp.opt_param_keys(orig_params):
        opt_params_only[k] = orig_params[k]
    opt_state = opt.init_fn(opt_params_only)
    print(f"...done")

    best_loss = float('inf')
    best_params = None



    @jax.jit
    def generate_batch(rngkey, samples_in, samples_out, samples_weight):

        # concatenate to make processing easier
        samples = jnp.concatenate((samples_in, samples_out[:,None], samples_weight[:,None]), axis=-1)

        # shuffle
        samples = jax.random.permutation(rngkey, samples, axis=0)

        # split in to batches
        # (discard any extra samples)
        batch_count = samples.shape[0] // args.batch_size
        n_batch_total = args.batch_size * batch_count
        samples = samples[:n_batch_total, :]

        # split back up
        samples_in = samples[:,:3]
        samples_out = samples[:,3]
        samples_weight = samples[:,4]

        batch_in = jnp.reshape(samples_in, (batch_count, args.batch_size, 3))
        batch_out = jnp.reshape(samples_out, (batch_count, args.batch_size))
        batch_weight = jnp.reshape(samples_weight, (batch_count, args.batch_size))

        return batch_in, batch_out, batch_weight, batch_count
    
    def batch_loss_fn(params, batch_coords, batch_target, batch_weight):

        add_full_params(params)
   
        def loss_one(params, coords, target, weight):
            pred = implicit_func(params, coords)

            if args.fit_mode == 'occupancy':
                return binary_cross_entropy_loss(pred, target)
            elif args.fit_mode == 'sdf':
                #L1 sdf loss
                return jnp.abs(pred - target)
            else: raise ValueError("bad arg")
        
        loss_terms = jax.vmap(partial(loss_one, params))(batch_coords, batch_target, batch_weight)
        loss_sum = jnp.mean(loss_terms)
        return loss_sum

    def batch_count_correct(params, batch_coords, batch_target):

        add_full_params(params)
   
        def loss_one(params, coords, target):
            pred = implicit_func(params, coords)

            if args.fit_mode == 'occupancy':
                is_correct_sign = jnp.sign(pred) == jnp.sign(target - .5)
                return is_correct_sign
            elif args.fit_mode in ['sdf']:
                is_correct_sign = jnp.sign(pred) == jnp.sign(target)
                return is_correct_sign
            else: raise ValueError("bad arg")
        
        correct_sign = jax.vmap(partial(loss_one, params))(batch_coords, batch_target)
        correct_count = jnp.sum(correct_sign)
        return correct_count

    @jax.jit
    def train_step(i_epoch, i_step, opt_state, batch_in, batch_out, batch_weight):
   
        opt_params = opt.params_fn(opt_state)
        value, grads = jax.value_and_grad(batch_loss_fn)(opt_params, batch_in, batch_out, batch_weight)
        correct_count = batch_count_correct(opt_params, batch_in, batch_out)
        opt_state = opt.update_fn(i_epoch, grads, opt_state)
        
        return value, opt_state, correct_count

    print(f"Training...")
    i_step = 0
    for i_epoch in range(args.n_epochs):
        
        key, subkey = jax.random.split(key)
        batches_in, batches_out, batches_weight, n_batches = generate_batch(subkey, samp, samp_target, samp_weight)
        losses = []
        n_correct = 0
        n_total = 0

        for i_b in range(n_batches):

            loss, opt_state, correct_count = train_step(i_epoch, i_step, opt_state, batches_in[i_b,...], batches_out[i_b,...], batches_weight[i_b,...])

            loss = float(loss)
            correct_count = int(correct_count)
            losses.append(loss)
            n_correct += correct_count
            n_total += args.batch_size
            i_step += 1

        mean_loss = np.mean(np.array(losses))
        frac_correct= n_correct / n_total

        print(f"== Epoch {i_epoch} / {args.n_epochs}   loss: {mean_loss:.6f}  correct sign: {100*frac_correct:.2f}%")

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = opt.params_fn(opt_state)
            add_full_params(best_params)
            print("  --> new best")

            print(f"Saving result to {args.output_file}")
            mlp.save(args.output_file, best_params)
            print(f"  ...done")

    
    # save the result
    print(f"Saving result to {args.output_file}")
    mlp.save(args.output_file, best_params)
    print(f"  ...done")


if __name__ == '__main__':
    main()
