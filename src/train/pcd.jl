"""
    pcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence.
"""
function pcd!(
    rbm::RBM,
    data::AbstractArray;
    batchsize::Int = 1,
    iters::Int = 1, # number of gradient updates
    wts::Union{AbstractVector, Nothing} = nothing, # data weights
    steps::Int = 1, # MC steps to update fantasy chains
    optim::AbstractRule = Adam(), # optimizer rule
    moments = moments_from_samples(rbm.visible, data; wts), # sufficient statistics for visible layer

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # gauge
    zerosum::Bool = true, # zerosum gauge for Potts layers
    rescale::Bool = true, # normalize continuous hidden units to var(h) = 1

    # momentum for hidden activity statistics tracking
    ρh::Real = 99//100,
    ϵh::Real = 1//100, # prevent vanishing var(h) estimate

    callback = Returns(nothing), # called for every batch

    # init fantasy chains
    vm = sample_from_inputs(rbm.visible, falses(size(rbm.visible)..., batchsize)),

    shuffle::Bool = true
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || size(data)[end] == length(wts)
    @assert ϵh ≥ 0

    # used to scale hidden unit activities
    var_h = total_var_from_inputs(rbm.hidden, inputs_h_from_v(rbm, data); wts)
    @assert all(var_h .+ ϵh .> 0)

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_hidden!(rbm, sqrt.(var_h .+ ϵh))

    # store average weight of each data point
    wts_mean = isnothing(wts) ? 1 : mean(wts)

    # define parameters for Optimisers
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w)
    state = setup(optim, ps)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        # update fantasy chains
        vm = sample_v_from_v(rbm, vm; steps)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        # correct weighted minibatch bias
        batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
        ∂ *= batch_weight

        # Exponential moving average of variance of hidden unit activations.
        ρh_eff = ρh ^ batch_weight # effective damp after 'batch_weight' updates
        var_h_batch = grad2var(rbm.hidden, -∂d.hidden) # extract hidden unit statistics from gradient
        var_h .= ρh_eff * var_h .+ (1 - ρh_eff) * var_h_batch
        @assert all(var_h .+ ϵh .> 0)

        # reset gauge
        rescale && rescale_hidden!(rbm, sqrt.(var_h .+ ϵh))
        zerosum && zerosum!(rbm)

        callback(; rbm, optim, iter, vm, vd, wd)
    end
    return rbm
end
