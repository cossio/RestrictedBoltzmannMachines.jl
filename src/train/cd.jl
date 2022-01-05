"""
    cd!(rbm, data)

Trains the RBM on data using contrastive divergence.
"""
function cd!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    lossadd = (_...) -> 0, # regularization
    verbose::Bool = true,
    ps = Flux.params(rbm),
    wts::Wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    for epoch in 1:epochs
        batches = minibatches(data, weights; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            # fantasy chains
            _idx = rand(1:_nobs(data), batchsize)
            _vm = selectdim(data, ndims(data), _idx)
            vm = sample_v_from_v(rbm, _vm; steps = steps)

            # compute contrastive divergence gradient
            gs = Zygote.gradient(ps) do
                loss = contrastive_divergence(rbm, vd, vm; wd)
                regu = lossadd(rbm, vd, vm, wd)
                ChainRulesCore.ignore_derivatives() do
                    push!(history, :cd_loss, loss)
                    push!(history, :reg_loss, regu)
                end
                return loss + regu
            end

            # update parameters using gradient
            Flux.update!(optimizer, ps, gs)

            push!(history, :epoch, epoch)
            push!(history, :batch, b)
        end

        lpl = batch_mean(log_pseudolikelihood(rbm, data), weights)
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

Contrastive divergence loss.
`vd` is a data sample, and `vm` are samples from the model.
"""
function contrastive_divergence(
    rbm::RBM, vd::AbstractTensor, vm::AbstractTensor; wd::Wts = nothing, wm::Wts = nothing
)
    Fd = mean_free_energy(rbm, vd; wts=wd)
    Fm = mean_free_energy(rbm, vm; wts=wm)
    return Fd - Fm
end

function mean_free_energy(
    rbm::RBM{<:AbstractLayer{N}}, v::AbstractTensor{N}; wts::Nothing = nothing
)::Real where {N}
    return free_energy(rbm, v)
end

function mean_free_energy(
    rbm::RBM, v::AbstractTensor{N}; wts::Wts = nothing
)::AbstractVector where {N}
    return batch_mean(free_energy(rbm, v))
end

function ∂contrastive_divergence(
    rbm::RBM, vd::AbstractTensor, vm::AbstractTensor; wd::Wts = nothing, wm::Wts = nothing
)
    ∂d = ∂free_energy(rbm, vd; wts = wd)
    ∂m = ∂free_energy(rbm, vm; wts = wm)
    @assert typeof(∂d) == typeof(∂m)
    return subtract_gradients(∂d, ∂m)
end

subtract_gradients(∂1::N, ∂2::N) where {N<:NamedTuple} = map(subtract_gradients, ∂1, ∂2)
subtract_gradients(∂1::A, ∂2::A) where {A<:AbstractTensor} = ∂1 - ∂2

function update!(rbm::RBM, ∂, optimizer)
    Flux.update!(optimizer, rbm.w, ∂.w)
    update!(rbm.visible, ∂.visible, optimizer)
    update!(rbm.hidden, ∂.hidden, optimizer)
end

function update!(layer::AbstractLayer, ∂, optimizer)
    for (k, g) in pairs(∂)
        Flux.update!(getproperty(layer, k), g, optimizer)
    end
end
