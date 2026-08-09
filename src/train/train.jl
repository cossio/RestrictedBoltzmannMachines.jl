#= Shared training loop for the `pcd!` trainers (plain, centered, and standardized).

The public trainers share the same skeleton: validate, prepare the data, set up the
optimizer, then repeatedly draw a minibatch, estimate the positive/negative phase
gradients, regularize, update the parameters, refresh model statistics and gauge
constraints, and report to the callback. They differ only in their negative-phase
estimator and in the model statistics and gauge operations they maintain, which each
wrapper supplies as closures.

Every trainer invokes its callback with the same keywords,
`(; rbm, optim, state, ps, iter, vd, wd, ∂)`, plus the extras returned by its
negative phase (`vm` for the PCD trainers). Callbacks should slurp unused keywords
with a trailing `_...`.

Gauge constraints are reset as `zerosum!` first, then rescaling: rescaling multiplies
the weights attached to each hidden unit by a scalar (or, for `StandardizedRBM`,
absorbs `scale_h` without touching the weights), which preserves the zero-sum gauge,
whereas `zerosum!` perturbs weight norms. In this order both constraints hold exactly
after each update. =#

function _train!(
        rbm, data::AbstractArray;
        batchsize::Int,
        iters::Int,
        wts::AbstractVector{<:Real},
        moments,
        optim::AbstractRule,
        ps, state,
        shuffle::Bool,
        l2_fields::Real, l1_weights::Real, l2_weights::Real, l2l1_weights::Real,
        zerosum::Bool,
        regularize::NamedTuple = (;), # extra ∂regularize! keywords (e.g. regularize_unstandardized)
        setup!, # setup!(data, wts): initial model statistics and gauge, after data preparation
        negative_phase, # negative_phase(vd) -> (∂m, extras); extras are forwarded to the callback
        post_update!, # post_update!(vd, wd, ∂d): model statistics updates and gauge reset
        callback,
    )
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    _validate_layer_parameters(rbm)
    isnothing(ps) && (ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w))
    isnothing(state) && (state = setup(optim, ps))

    batchsize > 0 || throw(ArgumentError("batchsize must be positive"))
    size(data, ndims(data)) > 0 ||
        throw(ArgumentError("data must contain at least one sample"))
    length(wts) == size(data, ndims(data)) ||
        throw(DimensionMismatch("length(wts) must equal the number of data samples"))
    _validate_weights(wts)
    wts_mean = mean(wts)
    batchsize = min(batchsize, length(wts))
    setup!(data, wts)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        # bias correction for the weighted minibatch, in the weights' own precision
        batch_weight = mean(wd) / wts_mean

        # positive and negative phase gradients
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m, extras = negative_phase(vd)
        # Correct the weighted minibatch bias, in the gradient eltype so the
        # correction scalar cannot promote narrow parameters. The correction
        # is at most the number of samples (the mean minibatch weight is at
        # most `length(wts) * wts_mean`), so the conversion cannot overflow
        # Float32 or wider.
        ∂ = (∂d - ∂m) * convert(float(real(eltype(∂d.w))), batch_weight)

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum, regularize...)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)
        _validate_layer_parameters(rbm)

        post_update!(vd, wd, ∂d)

        callback(; rbm, optim, state, ps, iter, vd, wd, ∂, extras...)
    end
    return state, ps
end

# default initialization of the persistent fantasy chains used by the PCD trainers
function _default_fantasy_chains(rbm, batchsize::Int)
    return sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize))
end

# negative phase shared by the PCD trainers: update the persistent chains in-place
function _pcd_negative_phase(rbm, vm, steps::Int)
    vm .= sample_v_from_v(rbm, vm; steps)
    return ∂free_energy(rbm, vm), (; vm)
end
