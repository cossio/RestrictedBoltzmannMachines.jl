"""
    cd!(rbm, data)

Trains the RBM on data using contrastive divergence.
"""
function cd!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optim = ADAM(), # optimizer algorithm
    wts = nothing, # weighted data points; named wts to avoid conflicts with RBM nomenclature
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    stats = suffstats(rbm, data; wts),
    callback = empty_callback
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        for (batch_idx, (vd, wd)) in enumerate(batches)
            vm = sample_v_from_v(rbm, vd; steps = steps)
            ∂ = ∂contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd, stats)
            update!(∂, rbm, optim)
            update!(rbm, ∂)
            callback(; rbm, optim, epoch, batch_idx, vm, vd, wd)
        end
    end
    return rbm
end

"""
    contrastive_divergence(rbm, vd, vm; wd = 1, wm = 1)

Contrastive divergence loss.
`vd` is a data sample, and `vm` are samples from the model.
"""
function contrastive_divergence(
    rbm::RBM, vd::AbstractArray, vm::AbstractArray; wd = nothing, wm = nothing
)
    Fd = mean_free_energy(rbm, vd; wts = wd)
    Fm = mean_free_energy(rbm, vm; wts = wm)
    return Fd - Fm
end

function mean_free_energy(rbm::RBM, v::AbstractArray; wts = nothing)::Number
    F = free_energy(rbm, v)
    if ndims(visible(rbm)) == ndims(v)
        wts::Nothing
        return F
    else
        return wmean(F; wts)
    end
end

function ∂contrastive_divergence(
    rbm::RBM, vd::AbstractArray, vm::AbstractArray;
    wd = nothing, wm = nothing,
    stats = suffstats(rbm, vd; wts = wd)
)
    ∂d = ∂free_energy(rbm, vd; wts = wd, stats)
    ∂m = ∂free_energy(rbm, vm; wts = wm)
    return subtract_gradients(∂d, ∂m)
end

subtract_gradients(∂1::NamedTuple, ∂2::NamedTuple) = map(subtract_gradients, ∂1, ∂2)
subtract_gradients(∂1::AbstractArray, ∂2::AbstractArray) where {N} = ∂1 - ∂2
