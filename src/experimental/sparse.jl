"""
    Experimental.Sparse

Experimental group-lasso weight regularization for structural (edge-level) sparsity on
Potts RBMs. Provides two `pcd!` variants — [`pcd_glasso!`](@ref) and [`pcd_gl2l1!`](@ref) —
that add a group-lasso penalty either as a subgradient term (default) or as a proximal
block-soft-threshold step (`prox = true`), the latter yielding *exact* zeros.

This API is experimental and may change without a breaking release.
"""
module Experimental

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
    # Training loops
    # ---------------------------------------------------------------------------------------

    # Shared PCD loop identical to the base `pcd!`, plus one extra group-lasso penalty applied
    # either as a subgradient (`sparse_grad!`, when `prox = false`) or as a proximal step
    # (`sparse_prox!`, when `prox = true`, after the optimizer update and before the gauge resets).
    function _pcd_sparse!(
            rbm::RBM, data::AbstractArray;
            sparse_grad!, sparse_prox!, prox::Bool, sparse_weight::Real, sparse_name::AbstractString,
            batchsize::Int = 1,
            iters::Int = 1,
            wts::Union{AbstractVector, Nothing} = nothing,
            steps::Int = 1,
            optim::AbstractRule = Adam(),
            moments = moments_from_samples(rbm.visible, data; wts),
            l2_fields::Real = 0,
            l1_weights::Real = 0,
            l2_weights::Real = 0,
            l2l1_weights::Real = 0,
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
        # The four weight-magnitude penalties (l1, l2l1, gl2l1, glasso) are overlapping sparsity
        # priors on the same weights; combining them is redundant, and mixing this penalty's
        # proximal step with another non-smooth penalty breaks the proximal-gradient objective.
        # Enforce that at most one is active (l2_weights, being smooth, is unrestricted).
        @assert count(!iszero, (l1_weights, l2l1_weights, sparse_weight)) ≤ 1 "at most one of l1_weights, l2l1_weights, and $sparse_name may be nonzero at a time"
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

            # base weight decay, then the group-lasso subgradient (unless applied proximally),
            # then project the full gradient onto the zerosum gauge
            ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)
            prox || sparse_grad!(∂, rbm)
            zerosum && zerosum!(∂, rbm)

            # feed gradient to Optimiser rule
            gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
            state, ps = update!(state, ps, gs)
            _validate_layer_parameters(rbm)

            # proximal group-lasso step, before the gauge resets (proximal-gradient order)
            prox && sparse_prox!(rbm)

            # reset gauge
            rescale && rescale_weights!(rbm)
            zerosum && zerosum!(rbm)

            callback(; rbm, optim, state, iter, vm, vd, wd)
        end
        return state, ps
    end

    """
        pcd_glasso!(rbm, data; glasso_weights = 0, prox = false, kwargs...)

    Train `rbm` with PCD plus a group-lasso (`glasso`) weight penalty
    `glasso_weights · ∑_edges ‖w[edge]‖₂` (block-`L2` over the Potts color axes). With
    `prox = false` (default) the penalty is added as a subgradient (like `l1_weights`); with
    `prox = true` it is applied as a proximal block-soft-threshold step ([`prox_glasso!`]) after
    each optimizer update, yielding *exact*, gauge-stable zeros. All other keyword arguments
    match the base [`RestrictedBoltzmannMachines.pcd!`](@ref).

    At most one of `l1_weights`, `l2l1_weights`, and `glasso_weights` may be nonzero — they are
    overlapping sparsity priors, and mixing the proximal step with another non-smooth penalty
    is ill-posed. `l2_weights` (smooth) is unrestricted.
    """
    function pcd_glasso!(rbm::RBM, data::AbstractArray; glasso_weights::Real = 0, prox::Bool = false, kwargs...)
        return _pcd_sparse!(
            rbm, data; prox, sparse_weight = glasso_weights, sparse_name = "glasso_weights",
            sparse_grad! = (∂, r) -> (iszero(glasso_weights) || (∂.w .+= ∂glasso_weights(r.w, glasso_weights, r)); nothing),
            sparse_prox! = r -> (iszero(glasso_weights) || prox_glasso!(r, glasso_weights); nothing),
            kwargs...,
        )
    end

    """
        pcd_gl2l1!(rbm, data; gl2l1_weights = 0, prox = false, kwargs...)

    Train `rbm` with PCD plus a group-`L2/L1` (`gl2l1`) weight penalty
    `gl2l1_weights · (1/2N) ∑_μ (∑_i ‖w[:, i, μ]‖₂)²` (the group version of `l2l1_weights`). With
    `prox = false` (default) the penalty is added as a subgradient (like `l2l1_weights`); with
    `prox = true` it is applied as the proximal step ([`prox_gl2l1!`], prox of the squared
    group-ℓ1,2 norm) after each optimizer update, yielding *exact* zeros. All other keyword
    arguments match the base [`RestrictedBoltzmannMachines.pcd!`](@ref).

    At most one of `l1_weights`, `l2l1_weights`, and `gl2l1_weights` may be nonzero — they are
    overlapping sparsity priors, and mixing the proximal step with another non-smooth penalty
    is ill-posed. `l2_weights` (smooth) is unrestricted.
    """
    function pcd_gl2l1!(rbm::RBM, data::AbstractArray; gl2l1_weights::Real = 0, prox::Bool = false, kwargs...)
        return _pcd_sparse!(
            rbm, data; prox, sparse_weight = gl2l1_weights, sparse_name = "gl2l1_weights",
            sparse_grad! = (∂, r) -> (iszero(gl2l1_weights) || (∂.w .+= ∂gl2l1_weights(r.w, gl2l1_weights, r.visible)); nothing),
            sparse_prox! = r -> (iszero(gl2l1_weights) || prox_gl2l1!(r, gl2l1_weights); nothing),
            kwargs...,
        )
    end

end # module Sparse

end # module Experimental
