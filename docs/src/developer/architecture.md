# Architecture

This page documents internal conventions of the package. It is aimed at
contributors; nothing here is needed to *use* the package.

## Array dimension convention

Data arrays have layer dimensions first and the batch dimension last. For a
layer of size `(N,)`, a single sample is a vector of length `N` and a batch is
a matrix of size `(N, B)`. For Potts layers of size `(Q, N)`, a batch has size
`(Q, N, B)`.

The weight array `w` has size `(size(visible)..., size(hidden)...)`. Functions
like `energy`, `free_energy` and `sample_from_inputs` follow this convention
and broadcast over trailing batch dimensions.

Internally, matrix products flatten the layer dimensions: `flat_w(rbm)`
reshapes `w` to `(length(visible), length(hidden))`, and `inputs_h_from_v`
computes `wflat' * vflat` with a single dense matrix multiplication (BLAS on
CPU, CUBLAS on GPU).

## Layer types and parameter storage

`AbstractLayer{N}` (where `N = ndims` of the layer) is the base type for all
unit types. Each layer implements: `energy`, `cgfs`, `sample_from_inputs`,
`mean_from_inputs`, `var_from_inputs`, `mode_from_inputs`.

All layers store their parameters in a single `.par` array. Its first
dimension indexes the parameters of the layer type (1 for `Binary` / `Spin` /
`Potts`, which only have `θ`; 2 for `Gaussian`, which has `θ` and `γ`; 4 for
`dReLU`). The remaining dimensions are the layer's spatial dimensions, so
`ndims(par) == N + 1`. Named parameter accessors such as `layer.θ` and
`layer.γ` are views into `.par`. Storing everything in one array makes
optimizer updates and GPU transfers simple (one array per layer).

`Potts` is special: it is an `AbstractLayer{2}` whose first dimension is the
one-hot (categorical) dimension with `Q` classes and whose second dimension is
the number of sites. So `size(potts_layer) == (Q, N)` and `par` has size
`(1, Q, N)`.

`RBM{V,H,W}` holds the `visible` layer, `hidden` layer and weights `w`. It is
extended by `CenteredRBM` (offsets) and `StandardizedRBM` (offsets and
scales).

## Module organization

- `src/layers/` — layer type definitions. `abstractlayer.jl` defines the
  interface; each file implements one layer type; `common.jl` has shared
  utilities.
- `src/rbms/` — convenience constructors (`BinaryRBM`, `HopfieldRBM`, ...).
- `src/train/` — training: `pcd.jl` (persistent contrastive divergence),
  `initialization.jl` (data-driven init), `gradient.jl`.
- `src/gauge/` — gauge transformations: `zerosum.jl`, `rescale_hidden.jl`,
  `shift_fields.jl`.
- `src/util/` — linear algebra helpers, one-hot encoding, truncated normal
  sampling.
- `ext/` — package extensions for CUDA (GPU support) and HDF5 (save / load).

The repository uses Julia workspaces (`[workspace]` in `Project.toml`) with
sub-projects: `test`, `docs`, `notebooks`, `repl`.
