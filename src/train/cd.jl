# Throw this from a callback to force an early stop of training
# (or just call stop())
struct EarlyStop <: Exception end
stop() = throw(EarlyStop())

"""
    unwhiten(rbm, data)

Given an RBM trained on whitened data, returns an RBM that can look at original data.
"""
function unwhiten(rbm::RBM, data::AbstractArray)

end

function whiten(data::AbstractArray)

end

struct WhiteningTransform

end

"""
    train!(rbm, data)

Trains the RBM on data.
"""
function train!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    opt = Flux.ADAM(), # optimizer algorithm
    ps::Flux.Params = Flux.params(rbm), # subset of optimized parameters
    history::MVHistory = MVHistory(), # stores training log
    callback = () -> (), # callback function called on each iteration
    lossadd = (_...) -> 0, # regularization
    verbose::Bool = true,
    weights::AbstractVector = trues(_nobs(data)), # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    white::Bool = false # input to hidden layer is whitened (similar to batch normalization)
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
                Fd = free_energy(rbm, vd)
                Fm = free_energy(rbm, vm)
                loss = weighted_mean(Fd, wd) - weighted_mean(Fm)
                regu = lossadd(rbm, vd, vm, wd)
                ChainRulesCore.ignore_derivatives() do
                    push!(history, :pcd_loss, loss)
                    push!(history, :reg_loss, regu)
                end
                return loss + regu
            end

            # update parameters using gradient
            Flux.update!(opt, ps, gs)

            push!(history, :epoch, epoch)
            push!(history, :batch, b)

            try
                callback()
            catch ex
                if ex isa EarlyStop
                    break
                else
                    rethrow(ex)
                end
            end
        end

        lpl = weighted_mean(log_pseudolikelihood(rbm, data), weights)
        push!(history, :lpl, lpl)
        if verbose
            Δt_ = round(Δt, digits=2)
            println("epoch $epoch/$epochs ($(Δt_)s), log(pseudolikelihood)=$lpl")
        end
    end
    return history
end
