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
    verbose::Bool = false,
    weights::AbstractVector = trues(_nobs(data)), # data point weights
    steps::Int = 1 # Monte Carlo steps to update fantasy particles
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert _nobs(data) == _nobs(weights)
    vm = sample_v_from_v(rbm, first(data).v, true, steps)
    progress_bar = Progress(minibatch_count(_nobs(data); batchsize = batchsize) * epochs)
    for epoch in 1:epochs
        for (vd, wd) in minibatches(data, weights; batchsize = batchsize)
            vm = update_chains(rbm, cd, vd, vm)
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

        #= record pseudo-likelihood over full dataset =#
        if !isnothing(history)
            lpl = log_pseudolikelihood_rand(rbm, data, 1, datum.w)
            push!(history, :lpl, iter, lpl)
            if iter % print_interval < data.batchsize
                println("iter=$iter, lpl_train=$lpl_train")
            end
            if !(lpl_train > min_lpl && lpl_tests > min_lpl)
                @error "lpl_train=$lpl_train or lpl_tests=$lpl_tests less than min_lpl=$min_lpl; stopping (iter=$iter)"
                throw(EarlyStop())
            end
        end
    end
    return nothing
end

function update_chains(rbm::RBM, vd::AbstractArray, vm::AbstractArray = vd, β::Real = 1)
    return sample_v_from_v(rbm, vm, β, cd.steps)::typeof(vm)
end

function update_chains(rbm::RBM, vd::AbstractArray, vm::AbstractArray = vd, β::Real = 1)
    return sample_v_from_v(rbm, vm, β, cd.steps)::typeof(vm)
end

"""
    contrastive_divergence(rbm, vd, vm, wd = 1, wm = 1)

Contrastive divergence, defined as free energy difference between data (vd) and
model sample (vm). The (optional) `wd, wm` are weights for the batches.
"""
function contrastive_divergence(rbm::RBM, vd, vm, wd = 1, wm = 1)
    Fd = mean_free_energy(rbm, vd, wd)
    Fm = mean_free_energy(rbm, vm, wm)
    return Fd - Fm
end

"""
    mean_free_energy(rbm, v, w = 1)

Mean free energy across visible configurations.
The optional `w` specifies weights for the data points.
"""
mean_free_energy(rbm::RBM, v::AbstractArray, w = 1) = weighted_mean(free_energy(rbm, v), w)
