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

    # optimization algorithm (computes parameter update step from gradients)
    optim = ADAM(),

    # fantasy chains
    vm = transfer_sample(visible(rbm), falses(size(visible(rbm))..., batchsize)),

    stats = suffstats(rbm, data; wts), # sufficient statistics for visible layer

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0,
    l2_weights::Real = 0,
    l2l1_weights::Real = 0,

    zerosum::Bool = true, # zerosum gauge for Potts layers

    # center gradients
    # for continuous hidden units, also scale hidden unit activities to var(h) = 1
    center::Bool = false, # center gradients
    damp::Real = 1e-3, # damping for hidden activity stats updates

    # tracks mean and variance of hidden unit activations
    λh = mean_h_from_v(rbm, data),
    νh = var_h_from_v(rbm, data),
    ϵh = 1e-2 # prevent vanishing var(h)
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # we center visible units with their average
    λv = batchmean(visible(rbm), data; wts)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # update fantasy chains
            vm .= sample_v_from_v(rbm, vm; steps)

            # contrastive divergence gradient
            ∂d = RBMs.∂free_energy(rbm, vd; wts = wd, stats)
            ∂m = RBMs.∂free_energy(rbm, vm)
            ∂ = RBMs.subtract_gradients(∂d, ∂m)
            push!(history, :∂, gradnorms(∂))

            if center
                have = grad2mean(hidden(rbm), ∂d.hidden)
                λh .= (1 - damp) * have .+ damp .* λh
                ∂ = center_gradient(rbm, ∂, λv, λh)
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
            if stdizeh
                hvar = grad2var(hidden(rbm), ∂d.hidden)
                νh .= (1 - damp) * hvar .+ damp .* νh
                rescale_hidden!(rbm, νh .+ ϵh)
            end
        end
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        push!(history, :vm, copy(vm))
        @debug "epoch $epoch/$epochs ($(round(Δt, digits=2))s)"
    end
    return history
end
