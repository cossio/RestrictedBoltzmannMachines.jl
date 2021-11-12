# Throw this from a callback to force an early stop of training
# (or just call stop())
struct EarlyStop <: Exception end
stop() = throw(EarlyStop())

"""
    train!(rbm, data)

Trains the RBM on data.
"""
function train!(rbm::RBM, data::AbstractArray;
    batchsize = 128,
    epochs = 100,
    opt = ADAM(), # optimizer algorithm
    ps::Params = params(rbm), # subset of optimized parameters
    history::MVHistory = MVHistory(), # stores training history
    callback = () -> (), # callback function called on each iteration
    lossadd = (_...) -> 0, # regularization
    verbose::Bool = true,
    weights::AbstractVector = trues(_nobs(data)), # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    batchnorm::Bool = true
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert _nobs(data) == _nobs(weights)

    # initialize fantasy chains
    _idx = randperm(_nobs(data))[1:batchsize]
    _vm = selectdim(data, ndims(data), _idx)
    vm = sample_v_from_v(rbm, _vm, β; steps = steps)

    # a nice progress bar to track training
    progress_bar = Progress(minibatch_count(_nobs(data); batchsize = batchsize) * epochs)

    for epoch in 1:epochs
        for (batch, vd, wd) in enumerate(minibatches(data, weights; batchsize = batchsize))
            # update fantasy chains
            vm = sample_v_from_v(rbm, vm, β; steps = steps)
            gs = gradient(ps) do
                loss = contrastive_divergence(rbm, vd, vm, wd)
                regu = lossadd(rbm, vd, vm, wd)
                ChainRulesCore.ignore_derivatives() do
                    push!(history, :pcd_loss, loss)
                    push!(history, :reg_loss, regu)
                end
                return loss + regu
            end
            Flux.update!(opt, ps, gs)

            push!(history, :epoch, epoch)
            push!(history, :batch, batch)

            try
                callback()
            catch ex
                if ex isa EarlyStop
                    break
                else
                    rethrow(ex)
                end
            end
            next!(progress_bar)
        end

        pl = weighted_mean(log_pseudolikelihood(rbm, data), weights)
        push!(history, :lpl, iter, pl)
        verbose && println("iter=$iter, log(pseudolikelihood)=$pl")
    end
    return rbm, history
end

function update_chains(
    rbm::RBM,
    vd::AbstractArray,
    vm::AbstractArray = vd,
    β::Real = 1;
    steps::Int = 1
)
    return sample_v_from_v(rbm, vm, β; steps = steps)::typeof(vm)
end

"""
    contrastive_divergence(rbm, vd, vm, wd = 1, wm = 1)

Contrastive divergence, defined as free energy difference between data (vd) and
model sample (vm). The (optional) `wd, wm` are weights for the batches.
"""
function contrastive_divergence(rbm::RBM, vd, vm, wd = 1, wm = 1)
    Fd = free_energy(rbm, vd)
    Fm = free_energy(rbm, vm)
    return weighted_mean(Fd, wd) - weighted_mean(Fm, wm)
    return Fd - Fm
end
