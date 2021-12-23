"""
    train!(rbm, data)

Trains the RBM on data.
"""
function train_white!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = Flux.ADAM(), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    lossadd = (_...) -> 0, # regularization
    verbose::Bool = true,
    weights::AbstractVector = trues(_nobs(data)), # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    initialize::Bool = false, # whether to initialize the RBM parameters
    whiten_v::Bool = false,
    whiten_ϵ::Real = 1e-6 # avoids singular cov matrix
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert _nobs(data) == _nobs(weights)

    if initialize
        initialize!(rbm, data)
    end

    # standardizes the data
    data_mat = reshape(data, length(rbm.visible), size(data)[end])
    data_mat_Σ = cov(data_mat; dims=2)
    whiten_μ = mean_(data_mat; dims=2)
    whiten_L = cholesky(data_mat_Σ + whiten_ϵ * I).L
    whiten_A = inv(whiten_L)
    data_mat_white = whiten_L \ (data_mat .- whiten_μ)
    data_mat_white = whiten_A * (data_mat .- whiten_μ)
    data_white = reshape(data_mat_white, size(data)...)

    # initialize fantasy chains
    _idx = rand(1:_nobs(data), batchsize)
    _vm = selectdim(data, ndims(data), _idx)
    vm = sample_v_from_v(rbm, _vm; steps = steps)

    if whiten_v
        data_ = data_white
    else
        data_ = data
    end

    for epoch in 1:epochs
        batches = minibatches(data_, weights; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            # update fantasy chains
            vm = sample_v_from_v(rbm, vm; steps = steps)

            if whiten_v
                rbm_ = whiten(rbm, whiten_L, whiten_μ)
            else
                rbm_ = rbm
            end
            ps = Flux.params(rbm_)

            # compute contrastive divergence gradient
            gs = Zygote.gradient(ps) do
                if weight_normalization
                    wl2 = sqrt.(sum(abs2, w_dirs; dims=layerdims(rbm.visible)))
                    rbm_norm = RBM(rbm.visible, rbm.hidden, w_norm .* w_dirs ./ wl2)
                    Fd = free_energy(rbm_norm, vd)
                    Fm = free_energy(rbm_norm, vm)
                else
                    Fd = free_energy(rbm_, vd)
                    Fm = free_energy(rbm_, vm)
                end
                loss = weighted_mean(Fd, wd) - weighted_mean(Fm)
                regu = lossadd(rbm_, vd, vm, wd)
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

            if whiten_v
                rbm = unwhiten(rbm_, whiten_L, whiten_μ)
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
    return rbm, history
end

function whiten(rbm::RBM{<:Binary, <:Binary}, L::LowerTriangular, μ::AbstractVector)
    # TODO: extend to other layers
    wmat = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    g = vec(rbm.visible.θ)
    θ = vec(rbm.hidden.θ)
    return RBM(
        Binary(reshape(L \ g, size(rbm.visible)...)),
        Binary(reshape(θ + wmat' * μ, size(rbm.hidden)...)),
        reshape(L \ wmat, size(rbm.weights)...)
    )
end

function unwhiten(rbm::RBM{<:Binary, <:Binary}, L::LowerTriangular, μ::AbstractVector)
    wmat = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    g = vec(rbm.visible.θ)
    θ = vec(rbm.hidden.θ)
    W = L * wmat
    return RBM(
        Binary(reshape(L * g, size(rbm.visible)...)),
        Binary(reshape(θ - W' * μ, size(rbm.hidden)...)),
        reshape(W, size(rbm.weights)...)
    )
end

function shift_field(layer::Union{Binary, Spin, Potts}, Δθ::AbstractArray)
    return typeof(layer)(layer.θ .+ Δθ)
end

function shift_field(layer::Union{Gaussian, ReLU}, Δθ::AbstractArray)
    return typeof(layer)(layer.θ .+ Δθ, layer.γ)
end

function shift_field(layer::dReLU, Δθ::AbstractArray)
    return typeof(layer)(layer.θp .+ Δθ, layer.θn .+ Δθ, layer.γp, layer.γn)
end
