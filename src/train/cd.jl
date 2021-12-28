"""
    train!(rbm, data)

Trains the RBM on data.
"""
function train!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    lossadd = (_...) -> 0, # regularization
    verbose::Bool = true,
    ps = Flux.params(rbm),
    weights::AbstractVector = trues(_nobs(data)), # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert _nobs(data) == _nobs(weights)

    # initialize fantasy chains
    _idx = rand(1:_nobs(data), batchsize)
    _vm = selectdim(data, ndims(data), _idx)
    vm = sample_v_from_v(rbm, _vm; steps = steps)

    for epoch in 1:epochs
        batches = minibatches(data, weights; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            # update fantasy chains
            vm = sample_v_from_v(rbm, vm; steps = steps)

            # compute contrastive divergence gradient
            gs = Zygote.gradient(ps) do
                loss = contrastive_divergence(rbm, vd, vm, wd)
                regu = lossadd(rbm, vd, vm, wd)
                ChainRulesCore.ignore_derivatives() do
                    push!(history, :pcd_loss, loss)
                    push!(history, :reg_loss, regu)
                end
                return loss + regu
            end

            # update parameters using gradient
            Flux.update!(optimizer, ps, gs)

            push!(history, :epoch, epoch)
            push!(history, :batch, b)
        end

        lpl = weighted_mean(log_pseudolikelihood(rbm, data), weights)
        push!(history, :lpl, lpl)
        if verbose
            Δt_ = round(Δt, digits=2)
            lpl_ = round(lpl, digits=2)
            println("epoch $epoch/$epochs ($(Δt_)s), log(pseudolikelihood)=$lpl_")
        end
    end
    return history
end

"""
    contrastive_divergence(rbm, vd, vm, wd = 1)

Contrastive divergence loss. `vd` is a data sample, and `vm` are samples from the model.
"""
function contrastive_divergence(rbm::RBM, vd, vm, wd = true)
    Fd = free_energy(rbm, vd)
    Fm = free_energy(rbm, vm)
    return weighted_mean(Fd, wd) - weighted_mean(Fm)
end

"""
    default_optimizer(nsamples, batchsize, epochs; opt = ADAM(), decay_after = 0.5)

The default optimizer decays the learning rate exponentially every epoch, starting after
`decay_after` of training time, with a pre-defined schedule.
"""
function default_optimizer(
    nsamples::Int, batchsize::Int, epochs::Int;
    opt = Flux.ADAM(), decay_final = 1e-2, decay_after = 0.5
)
    steps_per_epoch = minibatch_count(nsamples; batchsize = batchsize)
    nsteps = steps_per_epoch * epochs
    start = round(Int, nsteps * decay_after)

    decay = decay_final^inv(count((steps_per_epoch:steps_per_epoch:nsteps) .> start))
    return Flux.Optimise.Optimiser(opt, ExpDecay(1, decay, steps_per_epoch, decay_final, start))
end
