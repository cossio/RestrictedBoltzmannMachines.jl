"""
    train!(rbm, data)

Trains the RBM on data.
"""
function train!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = Flux.ADAM(), # optimizer algorithm
    ps::Flux.Params = Flux.params(rbm), # subset of optimized parameters
    history::MVHistory = MVHistory(), # stores training log
    callback = () -> (), # callback function called on each iteration
    lossadd = (_...) -> 0, # regularization
    verbose::Bool = true,
    weights::AbstractVector = trues(_nobs(data)), # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    weight_normalization::Bool = false, # https://arxiv.org/abs/1602.07868
    initialize::Bool = false, # whether to initialize the RBM parameters
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert _nobs(data) == _nobs(weights)

    if initialize
        initialize!(rbm, data)
    end

    if weight_normalization
        w_norm = sqrt.(sum(abs2, rbm.weights; dims=layerdims(rbm.visible)))
        w_dirs = copy(rbm.weights)
        ps = Flux.params(ps..., w_norm, w_dirs)
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
                if weight_normalization
                    wl2 = sqrt.(sum(abs2, w_dirs; dims=layerdims(rbm.visible)))
                    rbm_ = RBM(rbm.visible, rbm.hidden, w_norm .* w_dirs ./ wl2)
                    Fd = free_energy(rbm_, vd)
                    Fm = free_energy(rbm_, vm)
                else
                    Fd = free_energy(rbm, vd)
                    Fm = free_energy(rbm, vm)
                end
                loss = weighted_mean(Fd, wd) - weighted_mean(Fm)
                regu = lossadd(rbm, vd, vm, wd)
                ChainRulesCore.ignore_derivatives() do
                    push!(history, :pcd_loss, loss)
                    push!(history, :reg_loss, regu)
                end
                return loss + regu
            end

            # update parameters using gradient
            Flux.update!(optimizer, ps, gs)

            if weight_normalization
                # update RBM weights
                wl2 = sqrt.(sum(abs2, w_dirs; dims=layerdims(rbm.visible)))
                rbm.weights .= w_norm .* w_dirs ./ wl2
            end

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
