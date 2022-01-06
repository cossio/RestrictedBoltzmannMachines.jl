"""
    pcd_centered!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence, with centered gradients. See:

J. Melchior, A. Fischer, and L. Wiskott. JMLR 17.1 (2016): 3387-3447.
<http://jmlr.org/papers/v17/14-237.html>
"""
function pcd_centered!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    verbose::Bool = true,
    wts::Wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    α::Real = 0.5,
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # initialize fantasy chains
    _idx = rand(1:_nobs(data), batchsize)
    _vm = selectdim(data, ndims(data), _idx)
    vm = sample_v_from_v(rbm, _vm; steps = steps)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            # update fantasy chains
            vm = sample_v_from_v(rbm_, vm; steps = steps)

            # compute contrastive divergence gradient
            gs = Zygote.gradient(ps) do
                rbm_ = uncenter(rbm, λv, λh)
                loss = contrastive_divergence(rbm_, vd, vm; wd)
                regu = lossadd(rbm_, vd, vm, wd)
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

        lpl = batch_mean(log_pseudolikelihood(rbm_, data), wts)
        push!(history, :lpl, lpl)
        if verbose
            Δt_ = round(Δt, digits=2)
            lpl_ = round(lpl, digits=2)
            println("epoch $epoch/$epochs ($(Δt_)s), log(pseudolikelihood)=$lpl_")
        end
    end

    hdat = mean_h_from_v(rbm_, data)
    λv = mean_(data; dims=ndims(data))
    λh = mean_(hdat; dims=ndims(hdat))
    uncenter!(rbm, λv, λh)

    return history
end

# TODO: Implement for other layers
function ∂contrastive_divergence_centered(
    rbm::RBM{<:Binary, <:Binary}, vd::AbstractTensor, vm::AbstractTensor;
    wd::Wts = nothing, wm::Wts = nothing, ts = sufficient_statistics(rbm.visible, vd; wts)
)
    ∂d = ∂free_energy(rbm, vd; wts = wd, ts)
    ∂m = ∂free_energy(rbm, vm; wts = wm)
    ∂ = subtract_gradients(∂d, ∂m)

    # reuse estimates <h>d from the gradients
    λv = -∂d.visible.θ # uses full data thanks to sufficient_statistics mechanism
    λh = -∂d.hidden.θ # uses minibatch


end

function center_gradients!(∂::NamedTuple, rbm::RBM{<:Binary,<:Binary})
    μv = -∂.visible.θ
    μh = -∂.hidden.θ

    @assert size(∂w) == size(rbm.w)
    @assert size(λv) == size(rbm.visible)
    @assert size(λh) == size(rbm.hidden)

end
