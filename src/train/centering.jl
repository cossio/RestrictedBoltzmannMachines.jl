"""
    pcd_centered!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence, with centered gradients.
See:

J. Melchior, A. Fischer, and L. Wiskott. JMLR 17.1 (2016): 3387-3447.
<http://jmlr.org/papers/v17/14-237.html>
"""
function pcd_centered!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    verbose::Bool = true,
    wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    center_v::Bool = true, center_h::Bool = true, # center visible, hidden, or both
    α::Real = false # running average for λh
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # data statistics
    stats = sufficient_statistics(rbm.visible, data; wts)

    # initialize fantasy chains by sampling visible layer
    vm = transfer_sample(rbm.visible, falses(size(rbm.visible)..., batchsize))

    avg_h = batchmean(rbm.hidden, mean_h_from_v(rbm, data))

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # update fantasy chains
            vm = sample_v_from_v(rbm, vm; steps = steps)
            # compute centered gradients
            ∂ = ∂contrastive_divergence_centered(
                rbm, vd, vm; wd, stats, center_v, center_h, α, avg_h
            )
            # update parameters using gradient
            update!(optimizer, rbm, ∂)
            # store gradient norms
            push!(history, :∂, gradnorms(∂))
        end

        lpl = wmean(log_pseudolikelihood(rbm, data); wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        if verbose
            Δt_ = round(Δt, digits=2)
            lpl_ = round(lpl, digits=2)
            println("epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_")
        end
    end

    return history
end

# TODO: Implement for other layers
function ∂contrastive_divergence_centered(
    rbm::RBM, vd::AbstractArray, vm::AbstractArray;
    wd = nothing, wm = nothing,
    stats = sufficient_statistics(rbm.visible, vd; wts = wd),
    center_v::Bool = true, center_h::Bool = true,
    α::Real = false, avg_h::AbstractArray
)
    ∂d = ∂free_energy(rbm, vd; wts = wd, stats)
    ∂m = ∂free_energy(rbm, vm; wts = wm)
    ∂ = subtract_gradients(∂d, ∂m)

    # reuse moment estimates from the gradients
    λv = grad2mean(rbm.visible, ∂d.visible) # = <v>_d (uses full data, from sufficient_statistics)
    λh = grad2mean(rbm.hidden,  ∂d.hidden)  # = <h>_d (uses minibatch)

    # since h uses minibatches, keep running a average
    @assert size(avg_h) == size(λh)
    λh .= avg_h .= α * avg_h .+ (1 - α) .* λh

    ∂c = center_gradients(rbm, ∂, center_v * λv, center_h * λh)
    return ∂c
end

# estimate moments from gradients, e.g. <v> = -derivative of free energy w.r.t. θ
grad2mean(::Union{Binary,Spin,Potts,Gaussian,ReLU}, ∂::NamedTuple) = -∂.θ

function center_gradients(rbm::RBM, ∂::NamedTuple, λv::AbstractArray, λh::AbstractArray)
    @assert size(rbm.visible) == size(λv)
    @assert size(rbm.hidden) == size(λh)
    @assert size(∂.w) == size(rbm.w)

    ∂w_flat = reshape(∂.w, length(rbm.visible), length(rbm.hidden))
    ∂w_flat_c = ∂w_flat - vec(λv) * vec(∂.hidden.θ)' - vec(∂.visible.θ) * vec(λh)'
    ∂w_c = reshape(∂w_flat_c, size(rbm.w))

    shift_v = reshape(∂w_flat_c  * vec(λh), size(rbm.visible))
    shift_h = reshape(∂w_flat_c' * vec(λv), size(rbm.hidden))
    ∂v_c = center_gradients(rbm.visible, ∂.visible, shift_v)
    ∂h_c = center_gradients(rbm.hidden,  ∂.hidden,  shift_h)

    return (visible = ∂v_c, hidden = ∂h_c, w = ∂w_c)
end

function center_gradients(layer::Union{Binary,Spin,Potts}, ∂::NamedTuple, λ::AbstractArray)
    @assert size(layer) == size(∂.θ) == size(λ)
    return (; θ = ∂.θ - λ)
end

function center_gradients(layer::Union{Gaussian,ReLU}, ∂::NamedTuple, λ::AbstractArray)
    @assert size(layer) == size(∂.θ) == size(λ)
    return (θ = ∂.θ - λ, γ = ∂.γ)
end

function center_gradients(layer::dReLU, ∂::NamedTuple, λ::AbstractArray)
    @assert size(layer) == size(∂.θp) == size(∂.θn) == size(λ)
    return (θp = ∂.θp - λ, θn = ∂.θn - λ, γp = ∂.γp, γn = ∂.γn)
end

function center_gradients(layer::pReLU, ∂::NamedTuple, λ::AbstractArray)
    @assert size(layer) == size(∂.θ) == size(λ)
    return (θ = ∂.θ - λ, γ = ∂.γ, Δ = ∂.Δ, η = ∂.η)
end

function center_gradients(layer::xReLU, ∂::NamedTuple, λ::AbstractArray)
    @assert size(layer) == size(∂.θ) == size(λ)
    return (θ = ∂.θ - λ, γ = ∂.γ, Δ = ∂.Δ, ξ = ∂.ξ)
end
