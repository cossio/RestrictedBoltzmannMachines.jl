"""
    ∂regularize!(∂, rbm; l2_fields = 0, l1_weights = 0, l2_weights = 0, l2l1_weights = 0, gl2l1_weights = 0, glasso_weights = 0)

Updates RBM gradients `∂`, with the regularization gradient.

`gl2l1_weights` and `glasso_weights` are group-lasso penalties whose sparsity atom is a
block-`L2` norm over the Potts color axes of each site–hidden edge:

- `glasso` is the plain group lasso `∑_edges ‖w[edge]‖₂`, where each group spans the visible
  color axis (Potts visible) and the hidden color axis (Potts hidden); see [`prox_glasso!`].
- `gl2l1` is the group version of `l2l1`, `∑_μ (∑_i ‖w[:, i, μ]‖₂)² / (2N)` with `N` the
  number of visible sites, obtained by replacing the inner color-`L1` of `l2l1` with a
  color-`L2` group norm over the visible colors only (its hidden index `μ` is the outer
  `L2`-coupling axis).

With no Potts layers each `glasso` group is a single scalar, so `glasso` reduces to `l1`;
for a non-Potts visible layer `gl2l1` reduces to `l2l1`. For genuine edge sparsity, prefer
the proximal step [`prox_glasso!`](@ref) over adding the `glasso` subgradient here.
"""
function ∂regularize!(
        ∂::∂RBM, # unregularized gradient
        rbm::RBM;
        l2_fields::Real = 0, # L2 regularization of visible unit fields
        l1_weights::Real = 0, # L1 regularization of weights
        l2_weights::Real = 0, # L2 regularization of weights
        l2l1_weights::Real = 0, # L2/L1 regularziation of weights (10.7554/eLife.39397, Eq. 8)
        gl2l1_weights::Real = 0, # group L2/L1 regularization (color-L2 replacing the inner L1 of l2l1)
        glasso_weights::Real = 0, # group-lasso regularization (block-L2 over Potts colors)
        zerosum::Bool = false # whether to zerosum gradients
    )
    if !iszero(l2_fields)
        ∂regularize_fields!(∂.visible, rbm.visible; l2_fields)
    end
    if !iszero(l1_weights)
        ∂.w .+= l1_weights * sign.(rbm.w)
    end
    if !iszero(l2_weights)
        ∂.w .+= l2_weights * rbm.w
    end
    if !iszero(l2l1_weights)
        dims = ntuple(identity, ndims(rbm.visible))
        ∂.w .+= l2l1_weights * sign.(rbm.w) .* mean(abs, rbm.w; dims)
    end
    if !iszero(gl2l1_weights)
        ∂.w .+= ∂gl2l1_weights(rbm.w, gl2l1_weights, rbm.visible)
    end
    if !iszero(glasso_weights)
        ∂.w .+= ∂glasso_weights(rbm.w, glasso_weights, rbm)
    end
    zerosum && zerosum!(∂, rbm)
    return ∂
end

# Dimensions of the block-L2 group for group-lasso penalties: the visible color axis
# (dim 1) for Potts layers, and an empty (scalar) group for other visible layers, in which
# case group lasso reduces to L1.
_glasso_group_dims(::AbstractLayer) = ()
_glasso_group_dims(::Union{Potts, PottsGumbel}) = (1,)

# Number of block groups per hidden unit summed over in the gl2l1 inner sum: the number of
# visible sites for Potts (colors collapsed into each group norm), or the number of visible
# units for non-Potts layers (where each group is a scalar). Used to normalize gl2l1 so its
# strength does not scale with the number of Potts colors Q (and so gl2l1 reduces to l2l1).
function _glasso_num_groups(visible::AbstractLayer)
    dims = _glasso_group_dims(visible)
    return length(visible) ÷ prod(d -> size(visible, d), dims; init = 1)
end

# Block dimensions of the `glasso` atom over the full weight tensor `w`: the visible color
# axis (dim 1) when the visible layer is Potts, together with the hidden color axis
# (dim `ndims(visible)+1`) when the hidden layer is Potts. Grouping over the hidden color
# axis too keeps `prox_glasso!`'s exact zeros stable under the hidden Potts zero-sum gauge:
# `zerosum!` recenters each weight across sibling hidden colors, which would otherwise refill
# a slice zeroed over the visible colors only. Reduces to `()` (plain L1) when neither layer
# is Potts. `gl2l1` keeps the visible-only grouping (`_glasso_group_dims`) since its hidden
# axis is the outer L2-coupling index, not part of the block norm.
function _glasso_dims(rbm::RBM)
    vdims = _glasso_group_dims(rbm.visible)
    hdims = rbm.hidden isa Union{Potts, PottsGumbel} ? (ndims(rbm.visible) + 1,) : ()
    return (vdims..., hdims...)
end

# Group-`L2` norm with a zero-safe reverse-mode adjoint (∂‖w_g‖/∂w_g = w_g/‖w_g‖, taken to
# be `0` at `w_g = 0`, matching the `_inv_or_one` subgradient used by `∂glasso_weights` and
# `∂gl2l1_weights`). Plain `sqrt.(sum(abs2, w; dims))` is singular at zero groups under
# Zygote, which would otherwise poison the gradient of `regularization_penalty` with `NaN`s
# whenever the model has an exact-zero group (e.g. one already pruned by `prox_glasso!`).
_group_norm(w::AbstractArray; dims) = sqrt.(sum(abs2, w; dims))

function ChainRulesCore.rrule(::typeof(_group_norm), w::AbstractArray; dims)
    n = _group_norm(w; dims)
    function _group_norm_pullback(Δ)
        Δn = ChainRulesCore.unthunk(Δ)
        return ChainRulesCore.NoTangent(), Δn .* w .* _inv_or_one.(n)
    end
    return n, _group_norm_pullback
end

# gradient of glasso_weights * ∑_edges ‖w[edge]‖₂ (block-L2 over the Potts color axes)
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
    nrm = sqrt.(sum(abs2, w; dims)) # ‖w[:, i, μ]‖₂
    s = sum(nrm; dims = vdims)      # ∑_i ‖w[:, i, μ]‖₂ per hidden μ
    return (gl2l1_weights / N) * s .* w .* _inv_or_one.(nrm)
end

"""
    prox_glasso!(rbm, t; dims = <Potts color axes>)

In-place proximal (block soft-threshold) step for the group-lasso weight penalty
`t · ∑_edges ‖w[edge]‖₂`. By default each *edge* group spans the Potts color axes present:
the visible color axis when the visible layer is Potts, and the hidden color axis when the
hidden layer is Potts (for a non-Potts layer that axis contributes no grouping, and with no
Potts layers each group is a single scalar, i.e. an L1 soft-threshold). Each group is shrunk
toward zero,

    w[edge] .*= max(0, 1 - t / ‖w[edge]‖₂),

landing groups with `‖w[edge]‖₂ ≤ t` on *exact* zero. This reaches genuine edge-level
sparsity that a plain subgradient step does not.

descent. Because each group spans every Potts color axis and is scaled uniformly, the
resulting zeros are stable under the Potts zero-sum gauge (of *both* layers) and under
`rescale_weights!` — all preserve exact-zero groups — so the zeros persist for the callback
and the next fantasy update.

Unlike the gradient-based penalties (`l1_weights`, `l2_weights`, `l2l1_weights`,
`gl2l1_weights`), `t` is applied as a threshold directly and is *not* scaled by the
optimizer's step size: gradient-based penalties are folded into the gradient before
`optim` updates the parameters, so their effective strength automatically shrinks with
smaller learning rates, while `t` here always removes the same absolute amount of group
norm per update regardless of `optim`. Tune `glasso_weights` jointly with the chosen
`optim` (and its learning rate) rather than assuming it is directly comparable in scale to
the other `*_weights` regularizers.
"""
function prox_glasso!(rbm::RBM, t::Real; dims = _glasso_dims(rbm))
    nrm = sqrt.(sum(abs2, rbm.w; dims))
    rbm.w .*= max.(0, 1 .- t .* _inv_or_one.(nrm))
    return rbm
end

function ∂regularize_fields!(
        ∂::AbstractArray, layer::Union{Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, xReLU, pReLU, nsReLU}; l2_fields::Real = 0
    )
    if !iszero(l2_fields)
        ∂[1, ..] .+= l2_fields * layer.θ
    end
    return ∂
end

function ∂regularize_fields!(∂::AbstractArray, layer::dReLU; l2_fields::Real = 0)
    if !iszero(l2_fields)
        ∂[1, ..] .+= l2_fields * layer.θp
        ∂[2, ..] .+= l2_fields * layer.θn
    end
    return ∂
end

function ∂regularize(
        rbm::RBM;
        l2_fields::Real = 0, # L2 regularization of visible unit fields
        kw... # weights penalties
    )
    visible = ∂regularize_fields(rbm.visible; l2_fields)
    w = ∂regularize_weights(rbm; kw...)
    return ∂RBM(visible, zero(rbm.hidden.par), w)
end

function ∂regularize_fields(layer::Union{Binary, Spin, Potts}; l2_fields::Real = 0)
    ∂θ = l2_fields * layer.θ
    return vstack((∂θ,))
end

function ∂regularize_fields(layer::Union{Gaussian, ReLU}; l2_fields::Real = 0)
    ∂θ = l2_fields * layer.θ
    ∂γ = zero(layer.γ)
    return vstack((∂θ, ∂γ))
end

function ∂regularize_fields(layer::dReLU; l2_fields::Real = 0)
    ∂θp = l2_fields * layer.θp
    ∂θn = l2_fields * layer.θn
    ∂γn = zero(layer.γn)
    ∂γp = zero(layer.γp)
    return vstack((∂θp, ∂θn, ∂γp, ∂γn))
end

function ∂regularize_fields(layer::pReLU; l2_fields::Real = 0)
    ∂θ = l2_fields * layer.θ
    ∂γ = zero(layer.γ)
    ∂Δ = zero(layer.Δ)
    ∂η = zero(layer.η)
    return vstack((∂θ, ∂γ, ∂Δ, ∂η))
end

function ∂regularize_fields(layer::xReLU; l2_fields::Real = 0)
    ∂θ = l2_fields * layer.θ
    ∂γ = zero(layer.γ)
    ∂Δ = zero(layer.Δ)
    ∂ξ = zero(layer.ξ)
    return vstack((∂θ, ∂γ, ∂Δ, ∂ξ))
end

function ∂regularize_weights(
        rbm::RBM;
        l1_weights::Real = 0, # L1 regularization of weights
        l2_weights::Real = 0, # L2 regularization of weights
        l2l1_weights::Real = 0, # L2/L1 regularziation of weights (10.7554/eLife.39397, Eq. 8)
        gl2l1_weights::Real = 0, # group L2/L1 regularization
        glasso_weights::Real = 0 # group-lasso regularization
    )
    dims = ntuple(identity, ndims(rbm.visible))
    ∂l2l1 = l2l1_weights * sign.(rbm.w) .* mean(abs, rbm.w; dims)
    ∂l1 = l1_weights * sign.(rbm.w)
    ∂l2 = l2_weights * rbm.w
    ∂gl2l1 = ∂gl2l1_weights(rbm.w, gl2l1_weights, rbm.visible)
    ∂glasso = ∂glasso_weights(rbm.w, glasso_weights, rbm)
    return ∂l2l1 + ∂l1 + ∂l2 + ∂gl2l1 + ∂glasso
end

function regularization_penalty(
        rbm::RBM; l1_weights::Real = 0, l2_weights::Real = 0, l2l1_weights::Real = 0,
        gl2l1_weights::Real = 0, glasso_weights::Real = 0, l2_fields::Real = 0
    )
    dims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)
    Ng = _glasso_num_groups(rbm.visible) # number of sites (groups), = N for non-Potts

    # gl2l1 groups its inner block norm over the visible colors only (hidden is the outer
    # coupling axis); glasso groups over every Potts color axis (visible and hidden).
    nrm_gl2l1 = _group_norm(rbm.w; dims = _glasso_group_dims(rbm.visible))
    nrm_glasso = _group_norm(rbm.w; dims = _glasso_dims(rbm))

    reg_fields = l2_fields / 2 * regularization_penalty_fields(rbm.visible)
    reg_l1_weights = l1_weights * sum(abs, rbm.w)
    reg_l2_weights = l2_weights / 2 * sum(abs2, rbm.w)
    reg_l2l1_weights = l2l1_weights / (2N) * sum(abs2, sum(abs, rbm.w; dims))
    reg_gl2l1_weights = gl2l1_weights / (2Ng) * sum(abs2, sum(nrm_gl2l1; dims))
    reg_glasso_weights = glasso_weights * sum(nrm_glasso)

    return reg_fields + reg_l1_weights + reg_l2_weights + reg_l2l1_weights + reg_gl2l1_weights + reg_glasso_weights
end

regularization_penalty_fields(layer::dReLU) = sum(abs2, layer.θp) + sum(abs2, layer.θn)
regularization_penalty_fields(layer::Union{Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, pReLU, xReLU, nsReLU}) = sum(abs2, layer.θ)
