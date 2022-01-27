"""
    rdm!(rbm, data)

Trains the RBM on data using contrastive divergence with randomly initialized chains.
See http://arxiv.org/abs/2105.13889.
"""
function rdm!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = sufficient_statistics(rbm.visible, data; wts)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # fantasy particles, initialized randomly
            vm = randn(eltype(rbm.w), size(rbm.visible)..., batchsize)
            vm .= sample_v_from_v(rbm, vm; steps = steps)
            # compute gradients
            ∂ = ∂contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd, stats)
            # update parameters with gradients
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
