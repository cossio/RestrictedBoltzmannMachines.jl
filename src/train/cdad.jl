"""
    cdad!(rbm, data)

Trains the RBM on data using contrastive divergence.
Computes gradients with Zygote.
"""
function cdad!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optim = ADAM(),
    history::MVHistory = MVHistory(),
    wts = nothing,
    steps::Int = 1,
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            vm = sample_v_from_v(rbm, vd; steps = steps)
            gs = Zygote.gradient(rbm) do rbm
                contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd)
            end
            ∂ = only(gs)
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
