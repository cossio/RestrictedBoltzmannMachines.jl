"""
    rdm!(rbm, data)

Trains the RBM on data using contrastive divergence with randomly initialized chains.
See http://arxiv.org/abs/2105.13889.
"""
function rdm!(rbm::AbstractRBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optim = Flux.ADAM(),
    history::MVHistory = MVHistory(),
    wts = nothing,
    steps::Int = 1,
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = suffstats(visible(rbm), data; wts)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # fantasy particles, initialized randomly
            vm = transfer_sample(visible(rbm), FillArrays.Falses(size(visible(rbm)))..., batchsize)
            vm .= sample_v_from_v(rbm, vm; steps = steps)
            ∂ = ∂contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd, stats)
            push!(history, :∂, gradnorms(∂))
            update!(rbm, update!(∂, rbm, optim))
        end
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        @debug "epoch $epoch/$epochs ($(round(Δt, digits=2))s)"
    end
    return history
end
