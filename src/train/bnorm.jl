"""
    pcd_centered!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence, with centered gradients,
as done in the PGM repo, https://github.com/jertubiana/PGM.

This is almost equivalent to centering visible units only (see [`pcd_centered!`](@ref)),
but the centering of the hidden unit parameters is done much smoother.
"""
function pcd_bnorm!(rbm::RBM{<:Binary, <:Binary}, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = default_optimizer(_nobs(data), batchsize, epochs), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    verbose::Bool = true,
    wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
)
    @assert size(rbm.visible) == size(data)[1:ndims(rbm.visible)]
    # enforce one batch dimension, we need this for minibatching
    @assert ndims(data) == ndims(rbm.visible) + 1
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # data statistics
    stats = sufficient_statistics(rbm.visible, data; wts)

    avg_data = batchmean(rbm.visible, data)
    avg_inputs = inputs_v_to_h(rbm, avg_data)

    # initialize fantasy chains
    _idx = rand(1:_nobs(data), batchsize)
    _vm = selectdim(data, ndims(data), _idx)
    vm = sample_v_from_v(rbm, _vm; steps = steps)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # update batch norm reparameterization
            avg_inputs_new = inputs_v_to_h(rbm, avg_data)
            rbm.hidden.θ .-= avg_inputs_new .- avg_inputs
            avg_inputs = avg_inputs_new
            # update fantasy chains
            vm = sample_v_from_v(rbm, vm; steps = steps)
            # compute centered gradients
            ∂ = ∂contrastive_divergence_bnorm(rbm, vd, vm; wd, stats)
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
function ∂contrastive_divergence_bnorm(
    rbm::RBM{<:Binary, <:Binary}, vd::AbstractArray, vm::AbstractArray;
    wd = nothing, wm = nothing, stats
)
    ∂d = ∂free_energy(rbm, vd; wts = wd, stats)
    ∂m = ∂free_energy(rbm, vm; wts = wm)
    ∂ = subtract_gradients(∂d, ∂m)
    ∂w_flat = reshape(∂.w, length(rbm.visible), length(rbm.hidden))
    ∂w_center = ∂w_flat + vec(∂d.visible.θ) * vec(∂.hidden.θ)'
    return ∂c = (
        w = reshape(∂w_center, size(rbm.w)),
        visible = ∂.visible, hidden  = ∂.hidden
    )
end
