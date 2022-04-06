"""
    pcd_bnorm!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence, with centered gradients,
as done in the PGM repo, https://github.com/jertubiana/PGM.

This is almost equivalent to centering visible units only
(see <https://github.com/cossio/CenteredRBMs.jl>),
but the centering of the hidden unit parameters is done much smoother.
"""
function pcd_bnorm!(rbm::RBM{<:Binary, <:Binary}, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optim = default_optimizer(_nobs(data), batchsize, epochs),
    wts = nothing,
    steps::Int = 1,
)
    @assert size(visible(rbm)) == size(data)[1:ndims(visible(rbm))]
    @assert ndims(data) == ndims(visible(rbm)) + 1
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = suffstats(rbm, data; wts)

    avg_data = batchmean(visible(rbm), data)
    avg_inputs = inputs_v_to_h(rbm, avg_data)

    # initialize fantasy chains
    vm = selectdim(data, ndims(data), rand(1:_nobs(data), batchsize))
    vm = sample_v_from_v(rbm, vm; steps = steps)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        for (vd, wd) in batches
            # update batch norm reparameterization
            avg_inputs_new = inputs_v_to_h(rbm, avg_data)
            hidden(rbm).θ .-= avg_inputs_new .- avg_inputs
            avg_inputs = avg_inputs_new
            vm = sample_v_from_v(rbm, vm; steps = steps)
            ∂ = ∂contrastive_divergence_bnorm(rbm, vd, vm; wd, stats)
            update!(rbm, update!(∂, rbm, optim))
        end
    end
    return rbm
end

# TODO: Implement for other layers
function ∂contrastive_divergence_bnorm(
    rbm::RBM{<:Binary, <:Binary}, vd::AbstractArray, vm::AbstractArray;
    wd = nothing, wm = nothing, stats
)
    ∂d = ∂free_energy(rbm, vd; wts = wd, stats)
    ∂m = ∂free_energy(rbm, vm; wts = wm)
    ∂ = subtract_gradients(∂d, ∂m)
    ∂w_flat = reshape(∂.w, length(visible(rbm)), length(hidden(rbm)))
    ∂w_center = ∂w_flat + vec(∂d.visible.θ) * vec(∂.hidden.θ)'
    return ∂c = (
        w = reshape(∂w_center, size(rbm.w)),
        visible = ∂.visible, hidden  = ∂.hidden
    )
end
