"""
    train!(rbm, data)

Trains the RBM on data.
"""
function train!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = Flux.ADAM(), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    lossadd = (_...) -> 0, # regularization
    verbose::Bool = true,
    ps = Flux.params(rbm),
    weights::AbstractVector = trues(_nobs(data)), # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    initialize::Bool = false, # whether to initialize the RBM parameters
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert _nobs(data) == _nobs(weights)

    if initialize
        initialize!(rbm, data)
    end

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

            @show gs[rbm.weights]

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

function contrastive_divergence(rbm, vd, vm, wd)
    Fd = free_energy(rbm, vd)
    Fm = free_energy(rbm, vm)
    return weighted_mean(Fd, wd) - weighted_mean(Fm)
end
