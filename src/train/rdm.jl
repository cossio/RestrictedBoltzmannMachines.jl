"""
    rdm!(rbm, data)

Trains the RBM on data using contrastive divergence with randomly initialized chains.
See http://arxiv.org/abs/2105.13889.
"""
function rdm!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optim = Adam(),
    wts = nothing,
    steps::Int = 1,
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = suffstats(rbm.visible, data; wts)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        for (vd, wd) in batches
            # fantasy particles, initialized randomly
            vm = sample_from_inputs(rbm.visible, Falses(size(rbm.visible))..., batchsize)
            vm = sample_v_from_v(rbm, vm; steps = steps)
            ∂ = ∂contrastive_divergence(rbm, vd, vm; wd = wd, wm = wd, stats)
            update!(rbm, update!(∂, rbm, optim))
        end
    end
    return rbm
end
