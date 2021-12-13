"""
    train!(rbm, data)

Trains the RBM on data.
"""
function train!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = Flux.ADAM(), # optimizer algorithm
    ps::Flux.Params = Flux.params(rbm), # subset of optimized parameters
    history::MVHistory = MVHistory(), # stores training log
    callback = () -> (), # callback function called on each iteration
    lossadd = (_...) -> 0, # regularization
    verbose::Bool = true,
    weights::AbstractVector = trues(_nobs(data)), # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    initialize::Bool = false, # whether to initialize the RBM parameters
    weight_normalization::Bool = false, # https://arxiv.org/abs/1602.07868
    whiten_data::Bool = false, # whites v space. Similar https://jmlr.org/papers/volume17/14-237/14-237.pdf
    whiten_ϵ::Real = 1e-6, # avoids singular cov matrix
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert _nobs(data) == _nobs(weights)

    if initialize
        initialize!(rbm, data)
    end

    if weight_normalization
        w_norm = sqrt.(sum(abs2, rbm.weights; dims=layerdims(rbm.visible)))
        w_dirs = rbm.weights ./ w_norm
        ps = Flux.params(ps..., w_norm, w_dirs)
    end

    if whiten_data
        x = reshape(data, length(rbm.visible), size(data([end])))
        μ = mean(x; dims=2)
        C = cov(x; dims=2)
        L = cholesky(C + whiten_ϵ * I).L
        A = inv(cholesky(C + whiten_ϵ * I).L)
        x_white = A * (x - μ)
        data_white = reshape(x_white, size(data)...)
    end

    # initialize fantasy chains
    _idx = rand(1:_nobs(data), batchsize)
    _vm = selectdim(data, ndims(data), _idx)
    vm = sample_v_from_v(rbm, _vm; steps = steps)

    for epoch in 1:epochs
        batches = minibatches(data, weights; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            # update fantasy chains
            vm = sample_v_from_v(rbm, vm; steps = steps)

            # compute contrastive divergence gradient
            gs = Zygote.gradient(ps) do
                if weight_normalization
                    wl2 = sqrt.(sum(abs2, w_dirs; dims=layerdims(rbm.visible)))
                    rbm_ = RBM(rbm.visible, rbm.hidden, w_norm .* w_dirs ./ wl2)
                    Fd = free_energy(rbm_, vd)
                    Fm = free_energy(rbm_, vm)
                else
                    Fd = free_energy(rbm, vd)
                    Fm = free_energy(rbm, vm)
                end
                loss = weighted_mean(Fd, wd) - weighted_mean(Fm)
                regu = lossadd(rbm, vd, vm, wd)
                ChainRulesCore.ignore_derivatives() do
                    push!(history, :pcd_loss, loss)
                    push!(history, :reg_loss, regu)
                end
                return loss + regu
            end

            # update parameters using gradient
            Flux.update!(optimizer, ps, gs)

            if weight_normalization
                # update RBM weights
                wl2 = sqrt.(sum(abs2, w_dirs; dims=layerdims(rbm.visible)))
                rbm.weights .= w_norm .* w_dirs ./ wl2
            end

            push!(history, :epoch, epoch)
            push!(history, :batch, b)
        end

        lpl = weighted_mean(log_pseudolikelihood(rbm, data), weights)
        push!(history, :lpl, lpl)
        if verbose
            Δt_ = round(Δt, digits=2)
            lpl_ = round(lpl, digits=2)
            println("epoch $epoch/$epochs ($(Δt_)s), log(pseudolikelihood)=$lpl_")
        end
    end
    return history
end

function affine_rbm(rbm::RBM)

end


struct WhitenTransform{At<:AbstractMatrix, Bt::AbstractVector, iAt}
    whiten_A::At
    whiten_b::Bt
    invers_A::iAt
end

function WhitenTransform(data::AbstractMatrix, ϵ::Real = 1e-6)
    μ = mean(data; dims=2)
    C = cov(data; dims=2)
    L = cholesky(C + ϵ * I).L
    A = inv(cholesky(C + ϵ * I).L)
    return WhitenTransform(A, -A * μ, L)
end

function whiten_transform(t::WhitenTransform, data::AbstractMatrix)
    return t.whiten_A * data + t.whiten_b
end

function unwhiten_transform(t::WhitenTransform, wdat::AbstractArray)
    return
end

"""
    whiten_transform(rbm, transform)

Returns the whitened transform of `rbm`.
"""
function whiten_transform(
    transform::WhitenTransform, rbm::RBM{<:Union{Binary, Spin, Potts}}
)
    @assert size(transform.inverse_A) == (length(rbm.visible), length(rbm.visible))
    g = transform.inverse_A * reshape(rbm.visible.θ, length(rbm.visible))
    w = transform.inverse_A * reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    Δθ = -w' * transform.whitening_b
    return RBM(
        typeof(rbm.visible)(reshape(g, size(data.visible)...)),
        shift_field(rbm.hidden, reshape(Δθ, size(data.hidden)...)),
        reshape(w, size(rbm.visible)..., size(rbm.hidden)...)
    )
end

function unwhiten_transform(rbm::RBM, T::WhitenTransform)

end

function shift_field(layer::Union{Binary, Spin, Potts}, Δθ::AbstractArray)
    return typeof(layer)(layer.θ .+ shift)
end
