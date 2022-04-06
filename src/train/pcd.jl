"""
    pcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence.
"""
function pcd!(
    rbm::RBM,
    data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    wts = nothing, # data weights
    steps::Int = 1, # MC steps to update fantasy chains
    optim = default_optimizer(_nobs(data), batchsize, epochs), # optimization algorithm
    vm = fantasy_init(rbm, batchsize), # fantasy chains
    stats = suffstats(rbm, data; wts), # sufficient statistics for visible layer

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # gauge
    zerosum::Bool = true, # zerosum gauge for Potts layers
    center::Bool = true, # center gradients

    # scale hidden unit activations to var(h) = 1
    standardize_hidden::Bool = true,

    # damping for hidden activity statistics tracking
    hidden_damp::Real = batchsize / _nobs(data),
    ϵh = 1e-2, # prevent vanishing var(h)

    callback = empty_callback # called for every batch
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # we center layers with their average activities
    ave_v = batchmean(visible(rbm), data; wts)
    ave_h, var_h = meanvar_from_inputs(hidden(rbm), inputs_v_to_h(rbm, data); wts)

    # gauge constraints
    zerosum && zerosum!(rbm)
    standardize_hidden && rescale_hidden!(rbm, inv.(sqrt.(var_h .+ ϵh)))

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize)
        for (batch_idx, (vd, wd)) in enumerate(batches)
            # update fantasy chains
            vm .= sample_v_from_v(rbm, vm; steps)

            # contrastive divergence gradient
            ∂d = ∂free_energy(rbm, vd; wts = wd, stats)
            ∂m = ∂free_energy(rbm, vm)
            ∂ = subtract_gradients(∂d, ∂m)

            λh = grad2mean(hidden(rbm), ∂d.hidden)
            νh = grad2var(hidden(rbm), ∂d.hidden)
            ave_h .= (1 - hidden_damp) * λh .+ hidden_damp .* ave_h
            var_h .= (1 - hidden_damp) * νh .+ hidden_damp .* var_h

            if center
                ∂ = center_gradient(rbm, ∂, ave_v, ave_h)
            end

            # regularize
            ∂reg!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

            # compute parameter update step, according to optimizer algorithm
            update!(∂, rbm, optim)

            # get step in uncentered parameters
            if center
                ∂ = uncenter_step(rbm, ∂, ave_v, ave_h)
            end

            # update parameters with update step computed above
            update!(rbm, ∂)

            # respect gauge constraints
            zerosum && zerosum!(rbm)
            standardize_hidden && rescale_hidden!(rbm, inv.(sqrt.(var_h .+ ϵh)))

            callback(; rbm, optim, epoch, batch_idx, vm, vd, wd)
        end
    end
    return rbm
end

fantasy_init(rbm::RBM, sz) = transfer_sample(visible(rbm), falses(size(visible(rbm))..., sz))
empty_callback(@nospecialize(args...); @nospecialize(kw...)) = nothing
