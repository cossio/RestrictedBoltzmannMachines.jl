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
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # data statistics
    ts = sufficient_statistics(rbm.visible, data; wts)

    # initialize fantasy chains
    _idx = rand(1:_nobs(data), batchsize)
    _vm = selectdim(data, ndims(data), _idx)
    vm = sample_v_from_v(rbm, _vm; steps = steps)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # update fantasy chains
            vm = sample_v_from_v(rbm, vm; steps = steps)
            # compute centered gradients
            ∂c = ∂contrastive_divergence_centered(rbm, vd, vm; wd, ts)
            # update parameters using gradient
            update!(optimizer, rbm, ∂c)
        end

        lpl = batch_mean(log_pseudolikelihood(rbm, data), wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        if verbose
            Δt_ = round(Δt, digits=2)
            lpl_ = round(lpl, digits=2)
            println("epoch $epoch/$epochs ($(Δt_)s), log(pseudolikelihood)=$lpl_")
        end
    end

    return history
end

# TODO: Implement for other layers
function ∂contrastive_divergence_centered(
    rbm::RBM{<:Binary, <:Binary}, vd::AbstractTensor, vm::AbstractTensor;
    wd::Wts = nothing, wm::Wts = nothing, ts
)
    ∂d = ∂free_energy(rbm, vd; wts = wd, ts)
    ∂m = ∂free_energy(rbm, vm; wts = wm)
    ∂ = subtract_gradients(∂d, ∂m)

    # reuse estimates <h>d from the gradients
    λv = -∂d.visible.θ # uses full data thanks to sufficient_statistics mechanism
    λh = -∂d.hidden.θ # uses minibatch

    # flatten
    ∂w_flat = reshape(∂.w, length(rbm.visible), length(rbm.hidden))

    # centered gradients
    ∂w_c_flat = ∂w_flat - vec(λv) * vec(∂.hidden.θ)' - vec(∂.visible.θ) * vec(λh)'
    ∂v_c_flat = vec(∂.visible.θ) - ∂w_c_flat  * vec(λh)
    ∂h_c_flat = vec(∂.hidden.θ)  - ∂w_c_flat' * vec(λv)

    return ∂c = (
        w = reshape(∂w_c_flat, size(rbm.w)),
        visible = (; θ = reshape(∂v_c_flat, size(rbm.visible))),
        hidden  = (; θ = reshape(∂h_c_flat, size(rbm.hidden)))
    )
end
