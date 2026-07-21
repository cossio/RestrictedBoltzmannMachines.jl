"""
    Experimental.Sparse

Experimental group-lasso weight regularization for structural (edge-level) sparsity on
Potts RBMs. Provides a single training entry point [`proxpcd!`](@ref) that adds one
group-lasso penalty (`gl2l1` or `glasso`) either as a subgradient term (default) or as its
proximal operator (`prox = true`), the latter yielding *exact* zeros. The proximal operators
[`prox_gl2l1!`](@ref) and [`prox_glasso!`](@ref) are also exposed directly. (Plain elementwise
`l1`/`l2l1` are the non-Potts special cases of `glasso`/`gl2l1`.)

This API is experimental and may change without a breaking release.
"""
module Sparse

using ...RestrictedBoltzmannMachines: RBM, AbstractLayer, Potts, PottsGumbel
using ...RestrictedBoltzmannMachines: ∂free_energy, sample_v_from_v, sample_from_inputs,
    moments_from_samples, zerosum!, rescale_weights!, ∂regularize!, _validate_layer_parameters,
    _prepare_training_data, _batch_weight, infinite_minibatches, _inv_or_one
using Optimisers: AbstractRule, Adam, setup, update!
using FillArrays: Falses
import ChainRulesCore

# ---------------------------------------------------------------------------------------
# Group dimensions
# ---------------------------------------------------------------------------------------

# Block-L2 group over a single layer's color axis: dim 1 for Potts, empty (scalar) otherwise.
_glasso_group_dims(::AbstractLayer) = ()
_glasso_group_dims(::Union{Potts, PottsGumbel}) = (1,)

# Block dimensions of the `glasso` atom over the full weight tensor: the visible color axis
# (Potts visible) together with the hidden color axis (Potts hidden). Grouping over the
# hidden color axis too keeps prox zeros stable under the hidden Potts zero-sum gauge.
function _glasso_dims(rbm::RBM)
    vdims = _glasso_group_dims(rbm.visible)
    hdims = rbm.hidden isa Union{Potts, PottsGumbel} ? (ndims(rbm.visible) + 1,) : ()
    return (vdims..., hdims...)
end

# Number of visible sites (block groups per hidden unit) summed over in the gl2l1 inner sum:
# length(visible)/Q for Potts (colors collapsed into each group norm), length(visible)
# otherwise. Normalizes gl2l1 so its strength does not scale with the number of colors Q.
function _glasso_num_groups(visible::AbstractLayer)
    dims = _glasso_group_dims(visible)
    return length(visible) ÷ prod(d -> size(visible, d), dims; init = 1)
end

# ---------------------------------------------------------------------------------------
# Penalty and its (zero-safe) gradient
# ---------------------------------------------------------------------------------------

# Group-L2 norm with a zero-safe reverse-mode adjoint (∂‖w_g‖/∂w_g = w_g/‖w_g‖, taken as 0 at
# w_g = 0). Plain `sqrt.(sum(abs2, ·))` is singular at a zero group under Zygote.
_group_norm(w::AbstractArray; dims) = sqrt.(sum(abs2, w; dims))

function ChainRulesCore.rrule(::typeof(_group_norm), w::AbstractArray; dims)
    n = _group_norm(w; dims)
    function _group_norm_pullback(Δ)
        Δn = ChainRulesCore.unthunk(Δ)
        return ChainRulesCore.NoTangent(), Δn .* w .* _inv_or_one.(n)
    end
    return n, _group_norm_pullback
end

"""
    regularization_penalty(rbm; glasso_weights = 0, gl2l1_weights = 0)

Group-lasso weight penalties. `glasso` is the plain group lasso `∑_edges ‖w[edge]‖₂` (each
edge group spans the Potts color axes present, visible and hidden). `gl2l1` is the group
version of `l2l1`, `∑_μ (∑_i ‖w[:, i, μ]‖₂)² / (2N)` with `N` the number of visible sites,
grouping its inner block norm over the visible colors only.
"""
function regularization_penalty(rbm::RBM; glasso_weights::Real = 0, gl2l1_weights::Real = 0)
    dims = ntuple(identity, ndims(rbm.visible))
    Ng = _glasso_num_groups(rbm.visible)
    nrm_gl2l1 = _group_norm(rbm.w; dims = _glasso_group_dims(rbm.visible))
    nrm_glasso = _group_norm(rbm.w; dims = _glasso_dims(rbm))
    reg_glasso = glasso_weights * sum(nrm_glasso)
    reg_gl2l1 = gl2l1_weights / (2Ng) * sum(abs2, sum(nrm_gl2l1; dims))
    return reg_glasso + reg_gl2l1
end

# gradient of glasso_weights * ∑_edges ‖w[edge]‖₂
function ∂glasso_weights(w::AbstractArray, glasso_weights::Real, rbm::RBM)
    dims = _glasso_dims(rbm)
    nrm = sqrt.(sum(abs2, w; dims))
    return glasso_weights * w .* _inv_or_one.(nrm)
end

# gradient of gl2l1_weights / (2N) * ∑_μ (∑_i ‖w[:, i, μ]‖₂)², with N the number of sites
function ∂gl2l1_weights(w::AbstractArray, gl2l1_weights::Real, visible::AbstractLayer)
    dims = _glasso_group_dims(visible)
    vdims = ntuple(identity, ndims(visible))
    N = _glasso_num_groups(visible)
    nrm = sqrt.(sum(abs2, w; dims))
    s = sum(nrm; dims = vdims)
    return (gl2l1_weights / N) * s .* w .* _inv_or_one.(nrm)
end

# ---------------------------------------------------------------------------------------
# Proximal operators
# ---------------------------------------------------------------------------------------

"""
    prox_glasso!(rbm, t; dims = <Potts color axes>)

In-place proximal (block soft-threshold) step for `t · ∑_edges ‖w[edge]‖₂`. Each edge group
spans the Potts color axes present (visible and hidden); with no Potts layer each group is a
scalar (an L1 soft-threshold). Each group is shrunk toward zero,

    w[edge] .*= max(0, 1 - t / ‖w[edge]‖₂),

landing groups with `‖w[edge]‖₂ ≤ t` on *exact* zero. Because a group spans every Potts
color axis and is scaled uniformly, the zeros are stable under the Potts zero-sum gauge of
both layers and under `rescale_weights!`.
"""
function prox_glasso!(rbm::RBM, t::Real; dims = _glasso_dims(rbm))
    nrm = sqrt.(sum(abs2, rbm.w; dims))
    rbm.w .*= max.(0, 1 .- t .* _inv_or_one.(nrm))
    return rbm
end

# Threshold τ ≥ 0 for the prox of (c/2)·(∑ᵢ aᵢ)² over nonnegative `a` (the prox of the
# squared group-ℓ1,2 / squared-ℓ1 norm): the solution is βᵢ = max(0, aᵢ − τ) with the
# self-consistent τ = c·∑βᵢ, found by a sorted active-set scan.
function _prox_squared_l1_threshold(a::AbstractVector, c::Real)
    T = float(promote_type(eltype(a), typeof(c)))
    n = length(a)
    n == 0 && return zero(T)
    b = sort(a; rev = true)
    S = zero(T)
    τ = zero(T)
    for k in 1:n
        S += b[k]
        τk = c * S / (1 + c * k)
        if b[k] > τk
            τ = τk
        else
            break
        end
    end
    return τ
end

# Per-(site, hidden) shrink factors for the gl2l1 prox, from block norms `a` (already reduced
# over the visible color axis) with `N` visible sites and coupling strength `c`. Loops per
# hidden unit (a sort each) — CPU-oriented, unlike the elementwise glasso prox.
function _gl2l1_prox_factors(a::AbstractArray, N::Integer, c::Real)
    M = length(a) ÷ N
    a2 = reshape(a, N, M)
    factor = similar(a2, float(eltype(a2)))
    for μ in 1:M
        τ = _prox_squared_l1_threshold(view(a2, :, μ), c)
        for i in 1:N
            factor[i, μ] = max(zero(τ), one(τ) - τ * _inv_or_one(a2[i, μ]))
        end
    end
    return reshape(factor, size(a))
end

"""
    prox_gl2l1!(rbm, t; dims = <visible Potts color axis>)

In-place proximal step for the `gl2l1` penalty `t · (1/2N) · ∑_μ (∑_i ‖w[:, i, μ]‖₂)²` — the
prox of the squared group-ℓ1,2 norm. Unlike `glasso`, this couples the visible sites within
each hidden unit `μ`, so it is not a per-group soft-threshold: per hidden unit it soft-
thresholds the site block-norms with a *common* threshold `τ_μ = (t/N)·B_μ` set by the
surviving mass `B_μ` (a sorted active-set fixed point). Sites with `‖w[:, i, μ]‖₂ ≤ τ_μ` are
zeroed exactly. For a non-Potts visible layer (scalar groups) this is the prox of the
squared ℓ1 norm. CPU-oriented (a sort per hidden unit).

Note: the block norm is over the visible colors only (the hidden index `μ` is the outer
coupling axis), so for a Potts *hidden* layer with `zerosum = true` the subsequent
`zerosum!` can refill zeros across sibling hidden colors — persistent sparsity there needs
`zerosum = false`.
"""
function prox_gl2l1!(rbm::RBM, t::Real; dims = _glasso_group_dims(rbm.visible))
    N = _glasso_num_groups(rbm.visible)
    a = sqrt.(sum(abs2, rbm.w; dims))
    rbm.w .*= _gl2l1_prox_factors(a, N, t / N)
    return rbm
end

# ---------------------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------------------

# Apply the single active group-lasso subgradient (gl2l1 / glasso) to ∂.w.
function _add_sparse_subgradient!(∂, rbm, gl2l1_weights, glasso_weights)
    iszero(gl2l1_weights) || (∂.w .+= ∂gl2l1_weights(rbm.w, gl2l1_weights, rbm.visible))
    iszero(glasso_weights) || (∂.w .+= ∂glasso_weights(rbm.w, glasso_weights, rbm))
    return ∂
end

# Apply the single active penalty's proximal step (at most one weight is nonzero).
function _apply_sparse_prox!(rbm, gl2l1_weights, glasso_weights)
    iszero(gl2l1_weights) || prox_gl2l1!(rbm, gl2l1_weights)
    iszero(glasso_weights) || prox_glasso!(rbm, glasso_weights)
    return rbm
end

"""
    proxpcd!(rbm, data; prox = false, gl2l1_weights = 0, glasso_weights = 0,
             l2_weights = 0, l2_fields = 0, kwargs...)

Train `rbm` with PCD and a single group-lasso weight penalty, applied either as a
subgradient (`prox = false`, default) or as its proximal operator (`prox = true`, yielding
*exact* zeros). At most one of `gl2l1_weights` and `glasso_weights` may be nonzero — they
are overlapping sparsity priors, and a proximal step is only well-posed for a single
non-smooth penalty. The smooth terms `l2_weights` and `l2_fields` are unrestricted and
always added to the gradient. (Plain `l1`/`l2l1` are the non-Potts special cases of
`glasso`/`gl2l1`, so they are covered by those.)

With `prox = true` the active penalty is applied via its prox after each optimizer update,
before the gauge resets ([`prox_gl2l1!`] or [`prox_glasso!`]); `glasso` groups over every
Potts color axis, so its zeros are gauge-stable under `zerosum!` and `rescale_weights!`. All
other keyword arguments (`batchsize`, `iters`, `optim`, `steps`, `zerosum`, `rescale`,
`wts`, `callback`, …) match the base [`RestrictedBoltzmannMachines.pcd!`](@ref), including
GPU-array support for `rbm.w` — with one exception: `gl2l1_weights` with `prox = true`
calls [`prox_gl2l1!`], which is CPU-only (its per-hidden-unit sorted threshold uses scalar
indexing and will error under `allowscalar(false)`); `glasso_weights` (either mode) and
`gl2l1_weights` with `prox = false` are elementwise/broadcast-only and GPU-compatible.
"""
function proxpcd!(
        rbm::RBM, data::AbstractArray;
        prox::Bool = false,
        batchsize::Int = 1,
        iters::Int = 1,
        wts::Union{AbstractVector, Nothing} = nothing,
        steps::Int = 1,
        optim::AbstractRule = Adam(),
        moments = moments_from_samples(rbm.visible, data; wts),
        l2_fields::Real = 0,
        l2_weights::Real = 0,
        gl2l1_weights::Real = 0,
        glasso_weights::Real = 0,
        zerosum::Bool = true,
        rescale::Bool = true,
        callback = Returns(nothing),
        vm = nothing,
        shuffle::Bool = true,
        ps = nothing,
        state = nothing,
    )
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || size(data)[end] == length(wts)
    # gl2l1 and glasso are overlapping sparsity priors on the same weights, and a proximal
    # step is well-posed only for a single non-smooth penalty (prox(f+g) ≠ prox(f)∘prox(g)).
    # Enforce that at most one is active; l2_weights and l2_fields are smooth and unrestricted.
    @assert count(!iszero, (gl2l1_weights, glasso_weights)) ≤ 1 "at most one of gl2l1_weights and glasso_weights may be nonzero at a time"
    _validate_layer_parameters(rbm)
    isnothing(vm) && (vm = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)))
    isnothing(ps) && (ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w))
    isnothing(state) && (state = setup(optim, ps))

    data, wts, normalization, batchsize = _prepare_training_data(data, wts; batchsize)

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        batch_weight = _batch_weight(wd, normalization)

        # update fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        # correct weighted minibatch bias
        ∂ *= batch_weight

        # smooth weight decay, then the group-lasso subgradient (withheld when proxing —
        # applied after the update instead), then project onto the zerosum gauge
        ∂regularize!(∂, rbm; l2_fields, l2_weights)
        prox || _add_sparse_subgradient!(∂, rbm, gl2l1_weights, glasso_weights)
        zerosum && zerosum!(∂, rbm)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)
        _validate_layer_parameters(rbm)

        # proximal step, before the gauge resets (proximal-gradient order)
        prox && _apply_sparse_prox!(rbm, gl2l1_weights, glasso_weights)

        # reset gauge
        rescale && rescale_weights!(rbm)
        zerosum && zerosum!(rbm)

        callback(; rbm, optim, state, iter, vm, vd, wd)
    end
    return state, ps
end

end # module Sparse
