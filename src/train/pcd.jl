"""
    pcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence.
"""
function pcd!(
    rbm::RBM,
    data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    history::MVHistory = MVHistory(),
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

    # scale hidden unit activations to var(h) = 1 (requires center = true)
    standardize_hidden::Bool = true,

    hidden_damp::Real = 0.1 * batchsize / _nobs(data), # damping for hidden activity statistics tracking
    ϵh = 1e-2, # prevent vanishing var(h)

    callback = nothing # called for every batch
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    @assert center || !standardize_hidden

    # we center layers with their average activities
    ave_v = batchmean(visible(rbm), data; wts)
    ave_h, var_h = meanvar_from_inputs(hidden(rbm), inputs_v_to_h(rbm, data); wts)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize)
        Δt = @elapsed for (batch_idx, (vd, wd)) in enumerate(batches)
            # update fantasy chains
            vm .= sample_v_from_v(rbm, vm; steps)

            # contrastive divergence gradient
            ∂d = ∂free_energy(rbm, vd; wts = wd, stats)
            ∂m = ∂free_energy(rbm, vm)
            ∂ = subtract_gradients(∂d, ∂m)
            push!(history, :∂, gradnorms(∂))

            λh = grad2mean(hidden(rbm), ∂d.hidden)
            νh = grad2var(hidden(rbm), ∂d.hidden)
            ave_h .= (1 - hidden_damp) * λh .+ hidden_damp .* ave_h
            var_h .= (1 - hidden_damp) * νh .+ hidden_damp .* var_h

            if center
                ∂ = center_gradient(rbm, ∂, ave_v, ave_h)
            end

            # regularize
            ∂reg!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)
            push!(history, :∂r, gradnorms(∂))

            # compute parameter update step, according to optimizer algorithm
            update!(∂, rbm, optim)
            push!(history, :Δ, gradnorms(∂))

            # update parameters with update step computed above
            update!(rbm, ∂)

            # respect gauge constraints
            zerosum && zerosum!(rbm)
            standardize_hidden && rescale_hidden!(rbm, var_h .+ ϵh)

            isnothing(callback) || callback(; rbm, history, optim, epoch, batch_idx, vd, wd)
        end
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        push!(history, :vm, copy(vm))
        push!(history, :ave_h, copy(ave_h))
        push!(history, :var_h, copy(var_h))
        @debug "epoch $epoch/$epochs ($(round(Δt, digits=2))s)"
    end
    return history
end

fantasy_init(rbm::RBM, sz) = transfer_sample(visible(rbm), falses(size(visible(rbm))..., sz))
