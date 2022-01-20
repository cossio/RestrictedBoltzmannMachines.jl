"""
    pcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence.
"""
function pcd!(rbm::RBM, data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = sufficient_statistics(rbm.visible, data; wts)

    # initialize fantasy chains by sampling visible layer
    vm = transfer_sample(rbm.visible, falses(size(rbm.visible)..., batchsize))

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # update fantasy chains
            vm = sample_v_from_v(rbm, vm; steps = steps)
            # compute contrastive divergence gradient
            ∂ = ∂contrastive_divergence(rbm, vd, vm; wd, stats)
            # update parameters using gradient
            update!(optimizer, rbm, ∂)
            # store gradient norms
            push!(history, :∂, gradnorms(∂))
        end

        lpl = wmean(log_pseudolikelihood(rbm, data); wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)

        Δt_ = round(Δt, digits=2)
        lpl_ = round(lpl, digits=2)
        @debug "epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_"
    end
    return history
end
