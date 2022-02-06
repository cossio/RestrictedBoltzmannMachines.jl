"""
    cd!(rbm, data)

Trains the RBM on data using contrastive divergence.
"""
function cd!(rbm::AbstractRBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optim = Flux.ADAM(), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    wts = nothing, # weighted data points; named wts to avoid conflicts with RBM nomenclature
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    stats = suffstats(visible(rbm), data; wts)
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            vm = sample_v_from_v(rbm, vd; steps = steps)
            ∂ = ∂contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd, stats)
            push!(history, :∂, gradnorms(∂))
            update!(rbm, update!(∂, rbm, optim))
            push!(history, :Δ, gradnorms(∂))
        end
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        @debug "epoch $epoch/$epochs ($(round(Δt, digits=2))s)"
    end
    return history
end

"""
    contrastive_divergence(rbm, vd, vm; wd = 1, wm = 1)

Contrastive divergence loss.
`vd` is a data sample, and `vm` are samples from the model.
"""
function contrastive_divergence(
    rbm::AbstractRBM, vd::AbstractArray, vm::AbstractArray; wd = nothing, wm = nothing
)
    Fd = mean_free_energy(rbm, vd; wts = wd)
    Fm = mean_free_energy(rbm, vm; wts = wm)
    return Fd - Fm
end

function mean_free_energy(rbm::AbstractRBM, v::AbstractArray; wts = nothing)::Number
    F = free_energy(rbm, v)
    if ndims(visible(rbm)) == ndims(v)
        wts::Nothing
        return F
    else
        return wmean(F; wts)
    end
end

function ∂contrastive_divergence(
    rbm::AbstractRBM, vd::AbstractArray, vm::AbstractArray;
    wd = nothing, wm = nothing,
    stats = suffstats(visible(rbm), vd; wts = wd)
)
    ∂d = ∂free_energy(rbm, vd; wts = wd, stats)
    ∂m = ∂free_energy(rbm, vm; wts = wm)
    return subtract_gradients(∂d, ∂m)
end

subtract_gradients(∂1::NamedTuple, ∂2::NamedTuple) = map(subtract_gradients, ∂1, ∂2)
subtract_gradients(∂1::AbstractArray, ∂2::AbstractArray) where {N} = ∂1 - ∂2
