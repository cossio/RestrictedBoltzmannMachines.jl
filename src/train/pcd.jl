"""
    pcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence.
"""
function pcd!(
    rbm::RBM,
    data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    wts::Union{AbstractVector, Nothing} = nothing, # data weights
    steps::Int = 1, # MC steps to update fantasy chains
    optim = default_optimizer(_nobs(data), batchsize, epochs), # optimization algorithm
    stats = suffstats(rbm, data; wts), # sufficient statistics for visible layer

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # gauge
    zerosum::Bool = true, # zerosum gauge for Potts layers
    rescale::Bool = true, # normalize hidden units to var(h) = 1
    center::Bool = true, # center gradients

    # damping for hidden activity statistics tracking
    damp::Real = 1//100,
    ϵh::Real = 1e-2, # prevent vanishing var(h) estimate

    callback = empty_callback, # called for every batch
    mode::Symbol = :pcd, # :pcd, :cd, or :exact

    vm = fantasy_init(rbm; batchsize, mode), # fantasy chains
    shuffle::Bool = true
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || size(data)[end] == length(wts)
    @assert ϵh ≥ 0

    # we center layers with their average activities
    ave_v = batchmean(visible(rbm), data; wts)
    ave_h, var_h = total_meanvar_from_inputs(hidden(rbm), inputs_h_from_v(rbm, data); wts)
    @assert all(var_h .+ ϵh .> 0)

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_hidden!(rbm, sqrt.(var_h .+ ϵh))

    # store average weight of each data point
    wts_mean = isnothing(wts) ? 1 : mean(wts)

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize, shuffle)
        for (batch_idx, (vd, wd)) in enumerate(batches)
            # compute positive gradient from data
            ∂d = ∂free_energy(rbm, vd; wts = wd, stats)

            # compute negative gradient
            ∂m = ∂logpartition(rbm; vd, vm, wd, mode, steps)

            # likelihood gradient is the difference of positive and negative parts
            ∂ = subtract_gradients(∂d, ∂m)

            # correct weighted minibatch bias
            batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
            ∂ = gradmult(∂, batch_weight)

            # extract hidden unit statistics from gradient
            ave_h_batch = grad2mean(rbm.hidden, ∂d.hidden)
            var_h_batch = grad2var(rbm.hidden, ∂d.hidden)

            #= Exponential moving average of mean and variance of hidden unit activations.
            The batchweight can be interpreted as an "effective number of updates". =#
            damp_eff = damp ^ batch_weight # effective damp after 'batch_weight' updates
            ave_h .= (1 - damp_eff) * ave_h_batch .+ damp_eff * ave_h
            var_h .= (1 - damp_eff) * var_h_batch .+ damp_eff * var_h
            @assert all(var_h .+ ϵh .> 0)

            # weight decay
            ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

            if center
                # gradient of the centered RBM (Melchior et al 2016)
                ∂ = center_gradient(rbm, ∂, ave_v, ave_h)
            end

            # compute parameter update step from gradient, according to optimizer algorithm
            update!(∂, rbm, optim)

            if center
                # transform step in centered coordinates to uncentered coordinates
                ∂ = uncenter_step(rbm, ∂, ave_v, ave_h)
            end

            # update parameters with update step computed above
            update!(rbm, ∂)

            # reset gauge
            zerosum && zerosum!(rbm)
            rescale && rescale_hidden!(rbm, sqrt.(var_h .+ ϵh))

            callback(; rbm, optim, epoch, batch_idx, vm, vd, wd)
        end
    end
    return rbm
end

# init fantasy chains
fantasy_init(rbm::RBM; batchsize::Int, mode::Symbol = :pcd) = fantasy_init(rbm.visible; batchsize, mode)

function fantasy_init(l::AbstractLayer; batchsize::Int, mode::Symbol = :pcd)
    @assert mode ∈ (:pcd, :cd, :exact)
    if mode === :pcd || mode === :cd
        return sample_from_inputs(l, falses(size(l)..., batchsize))
    elseif mode === :exact
        @warn "Running extensive sampling; this can take a lot of RAM and time"
        return extensive_sample(l)
    end
end

empty_callback(@nospecialize(args...); @nospecialize(kw...)) = nothing

function ∂logpartition(
    rbm::RBM;
    vd::AbstractArray,
    vm::AbstractArray,
    wd::Union{AbstractArray, Nothing},
    mode::Symbol,
    steps::Int
)
    @assert mode ∈ (:pcd, :cd, :exact)
    if mode === :pcd
        # update persistent fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)
        return ∂free_energy(rbm, vm)
    elseif mode === :cd
        # chains re-initialized from data
        vm .= sample_v_from_v(rbm, vd; steps)
        return ∂free_energy(rbm, vm; wts = wd)
    elseif mode === :exact
        # use exact RBM probabilities (assuming `vm` is an extensive sample)
        p = softmax(-free_energy(rbm, vm))
        return ∂free_energy(rbm, vm; wts = p)
    end
end

function extensive_sample(layer::Binary; maxlen::Int = 12)
    @assert length(layer) ≤ maxlen
    N = length(layer)
    v = reduce(hcat, digits.(Bool, 0:(2^N - 1), base=2, pad=N))
    return reshape(v, size(layer)..., :)
end

function extensive_sample(layer::Spin; maxlen::Int = 12)
    σ = extensive_sample(Binary(layer.θ); maxlen)
    return Int8(2) * σ .- Int8(1)
end

function extensive_sample(layer::Potts; maxlen::Int = 12)
    q = size(layer, 1)
    N = prod(tail(size(layer)))
    @assert N * log2(q) ≤ maxlen && q < typemax(Int8)
    potts = reduce(hcat, digits.(Int8, 0:(q^N - 1), base=q, pad=N))
    onehot = reshape(potts, 1, :) .== 0:(q - 1)
    return reshape(onehot, size(layer)..., :)
end
