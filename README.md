Perform geometric queries on **neural implicit surfaces** like **ray casting**, **intersection testing**, **fast mesh extraction**, **closest points**, and **more**. Works on _general_ neural implicit surfaces (i.e. does not require a fitted signed distance function). Implemented in JAX.

<p align="center">
<img src="https://github.com/nmwsharp/neural-implicit-queries/blob/main/assets/images/spelunking_teaser_big.jpg" width="600"> 
</p>
<p align="center"> 
  <a href=https://nmwsharp.com/research/interval-implicits/>[project page]</a> &nbsp;&nbsp; 
  <a href=https://nmwsharp.com/media/papers/interval-implicits/SpelunkingTheDeep.pdf>[PDF (4MB)]</a>  &nbsp;&nbsp;
  <b>Authors:</b> <a href=https://nmwsharp.com/>Nicholas Sharp</a> & <a href=https://www.cs.toronto.edu/~jacobson/>Alec Jacobson</a>
<p align="center">


This code accompanies the paper **"Spelunking the Deep: Guaranteed Queries for General Neural Implicit Surfaces via Range Analysis"**, published at **SIGGRAPH 2022** (recognized with a Best Paper Award!).

---

Neural implicit surface representations encode a 3D surface as a level set of a neural network applied to coordinates; this representation has many promising properties. But once you have one of these surfaces, how do you perform standard geometric queries like casting rays against the surface, or testing if two such surfaces intersect? This is especially tricky if the neural function is _not_ a signed distance function (SDF), such as occupancy functions as in popular radiance field formulations, or randomly initialized networks during training.

This project introduces a technique for implementing these queries using __range analysis__, and automatic function transformation which we use to analyze a forward pass of the network and compute provably-valid bounds on the output range of the network over a spatial region. This basic operation is used as a building block for a variety of geometric queries.


## How does it work? How do I use it?

This project **does not** propose any new network architectures, training procedures, etc. Instead, it takes an existing neural implicit MLP and analyzes it to perform geometric queries. Right now the code is only set up for simple feedforward MLP architectures with ReLU or TanH activations (please file an issue to tell us about other architectures you would like to see!).

Exposing this machinery as a library is tricky, because the algorithm needs to analyze the internals of your neural network evaluation to work (somewhat similar to autodiff frameworks). For this reason, the library takes a simple specification of your MLP in a dictionary format; a convenience script is included to fit MLPs in this format, or see [below](TODO) for manually constructing dictionaries from your own data.

Once your MLP is ready in the format expected by this library, the functions in `queries.py` (raycasting) and `kd_tree.py` (spatial queries) can be called to perform queries. 

Additionally, several [demo scripts](#demo-scripts) are included to interactively explore these queries.


## Quick guide:

- Affine arithmetic rules appear in `affine.py` and `affine_layers.py`
- Queries are implemented in `queries.py` (raycasting) and `kd_tree.py` (spatial queries)

> **PERFORMANCE NOTE:** JAX uses JIT-compiled kernels for performance. All algorithms will be dramatically slower on the first pass due to JIT compilation (which can take up to a minute). We make use of bucketing to ensure there are only a small number of kernels that need to be JIT'd for a given algorithm, but it still takes time. all routines should be run twice to get an actual idea of performance.


## Installation

This code has been tested on both Linux and OSX machines. Windows support is unknown.

Some standard Python packages are needed, all available in package managers. A conda `environment.yml` file is included to help setting up the environment, but note that installing JAX may require nonstandard instructions---see the JAX documentation. Code has been tested with JAX 0.2.27 and 0.3.4.


## Demo scripts

#### Spelunking

This script provides an interactive GUI allowing the exploration of most of the queries described in this work.

Run like:

```
python src/main_spelunking.py sample_inputs/fox.npz
```

This application can run most of the algorithms described in this work. Use the buttons on the right to explore them and visualize the output.

Shapes are "previewed" via coarse meshes for the sake of the user interface. The coarse preview mesh is not used for any computation.

To make it easier to see hierarchy trees, enable a slice plane in upper left menu panel under [View] --> [Slice Plane].


#### Intersection

This script provides an interactive GUI allowing the exploration of intersection testing queries between two neural implicit shapes.

Run like:

```
python src/main_intersection.py sample_inputs/hammer.npz sample_inputs/bunny.npz
```

To adjust the objects, click in the left menu bar [Options] --> [Transform] --> [Show Gizmo] and drag around. Don't use the scaling function of the gizmo, it is not supported.

This query is configured to detect a single intersection, and exits as soon as any intersection is shown. The result will be printed to the terminal, and a point will be placed at the intersection location, though this location will be inside a shape, of course.

To make it easier to see intersections, try [Options] --> [Transparency] to make meshes transparent.

To make it easier to see hierarchy trees, enable a slice plane in upper left menu panel under [View] --> [Slice Plane].


#### Fit implicit

This script is a helper to quickly fit suitable test MLPs to triangle meshes and save them in the format expected by this codebase.

Run like:
```
python src/main_fit_implicit.py path/to/your/favorite/mesh.obj mlp_out.npz 
```

If you would like to fit your own implicit functions in our format, this is a simple script to fit an implicit function to a given mesh. The flags have options for sdf vs. occupancy, layer sizes, etc.



## Misc notes

Currently, JAX defaults to allocating nearly all GPU memory at startup. This may cause problems when subroutines external to JAX attempt to allocate additional memory. One workaround is to prepend the environment variable `XLA_PYTHON_CLIENT_MEM_FRACTION=.60`
