"""
    pcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence.
"""
function pcd!(rbm::RBM, data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optim = Flux.ADAM(),
    history::MVHistory = MVHistory(),
    wts = nothing,
    steps::Int = 1,
    vm = transfer_sample(visible(rbm), falses(size(visible(rbm))..., batchsize)), # fantasy chains
    stats = suffstats(rbm, data; wts) # sufficient statistics
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            vm = sample_v_from_v(rbm, vm; steps = steps)
            ∂ = ∂contrastive_divergence(rbm, vd, vm; wd, stats)
            push!(history, :∂, gradnorms(∂))
            update!(rbm, update!(∂, rbm, optim))
            push!(history, :Δ, gradnorms(∂))
        end
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        push!(history, :vm, copy(vm))
        @debug "epoch $epoch/$epochs ($(round(Δt, digits=2))s)"
    end
    return history
end
