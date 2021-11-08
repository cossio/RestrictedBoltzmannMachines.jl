# Throw this from a callback to force an early stop of training
# (or just call stop())
struct EarlyStop <: Exception end
stop() = throw(EarlyStop())

"""
    train!(rbm, data)
"""
function train!(rbm::RBM, data::AbstractArray;
    batchsize = 128,
    epochs = 100,
    opt = ADAM(), # optimizer algorithm
    ps::Params = params(rbm), # subset of optimized parameters
    history = MVHistory(), # stores training history
    callback = () -> (), # callback function called on each iteration
    lossadd = () -> 0, # regularization
    verbose::Bool = true,
    weights::AbstractVector = trues(_nobs(data)), # data point weights
    steps::Int = 1 # Monte Carlo steps to update fantasy particles
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert _nobs(data) == _nobs(weights)
    _vm = selectdim(data, ndims(data), randperm(_nobs(data))[1:batchsize])
    vm = sample_v_from_v(rbm, _vm, β; steps = steps)
    progress_bar = Progress(minibatch_count(_nobs(data); batchsize = batchsize) * epochs)
    for epoch in 1:epochs
        for (vd, wd) in minibatches(data, weights; batchsize = batchsize)
            vm = sample_v_from_v(rbm, vm, β; steps = steps)
            gs = gradient(ps) do
                L = contrastive_divergence(rbm, vd, vm, wd)
                R = lossadd()
                if !isnothing(history)
                    ignore() do
                        push!(history, :loss, iter, L)
                        push!(history, :regularization, iter, R)
                    end
                end
                return L + R
            end
            Flux.update!(opt, ps, gs)
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
