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
        wts::Union{AbstractVector, Nothing},
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
    @assert isnothing(wts) || size(data)[end] == length(wts)
    _validate_layer_parameters(rbm)
    isnothing(ps) && (ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w))
    isnothing(state) && (state = setup(optim, ps))

    data, wts, normalization, batchsize = _prepare_training_data(data, wts; batchsize)
    setup!(data, wts)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        batch_weight = _batch_weight(wd, normalization)

        # positive and negative phase gradients
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m, extras = negative_phase(vd)
        ∂ = (∂d - ∂m) * batch_weight # correct weighted minibatch bias

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
