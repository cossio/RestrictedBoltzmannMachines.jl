"""
    cdad!(rbm, data)

Trains the RBM on data using contrastive divergence.
Computes gradients with Zygote.
"""
function cdad!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optim = Adam(),
    wts = nothing,
    steps::Int = 1,
    callback = empty_callback
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        for (batch_idx, (vd, wd)) in enumerate(batches)
            vm = sample_v_from_v(rbm, vd; steps = steps)
            ∂, = Zygote.gradient(rbm) do rbm
                contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd)
            end
            update!(∂, rbm, optim)
            update!(rbm, ∂)
            callback(; rbm, optim, epoch, batch_idx, vm, vd, wd)
        end
    end
    return rbm
end
