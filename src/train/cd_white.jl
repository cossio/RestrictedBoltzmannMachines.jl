"""
    train_white!(rbm, data)

Trains the RBM on data.
"""
function train_white!(rbm::RBM{<:Binary, <:Binary}, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = Flux.ADAM(), # optimizer algorithm
    history::MVHistory = MVHistory(), # stores training log
    verbose::Bool = true,
    wts::Wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    initialize::Bool = false, # whether to initialize the RBM parameters
    whiten_ϵ::Real = 1e-6 # avoids singular cov matrix
)
    check_size(rbm.visible, data)
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    if initialize
        initialize!(rbm, data)
    end

    # standardizes the data
    data_mat = reshape(data, length(rbm.visible), size(data)[end])
    data_mat_Σ = cov(data_mat; dims=2)
    whiten_μ = mean_(data_mat; dims=2)
    whiten_L = cholesky(data_mat_Σ + whiten_ϵ * I).L
    data_mat_white = whiten_L \ (data_mat .- whiten_μ)
    data_white = reshape(data_mat_white, size(data)...)

    # initialize fantasy chains
    _idx = rand(1:_nobs(data), batchsize)
    _vm = selectdim(data, ndims(data), _idx)
    vm = sample_v_from_v(rbm, _vm; steps = steps)

    rbm_black = rbm

    for epoch in 1:epochs
        batches = minibatches(data_white, wts; batchsize = batchsize)
        Δt = @elapsed for (b, (vd, wd)) in enumerate(batches)
            # update fantasy chains
            vm = sample_v_from_v(rbm_black, vm; steps = steps)
            rbm_white = whiten(rbm_black, whiten_L, whiten_μ)
            vm_white = whiten(vm, whiten_L, whiten_μ)

            # compute contrastive divergence gradient
            ps = Flux.params(rbm_white)
            gs = Zygote.gradient(ps) do
                loss = contrastive_divergence(rbm_white, vd, vm_white; wd)
                ChainRulesCore.ignore_derivatives() do
                    push!(history, :cd_loss, loss)
                end
                return loss
            end
            # update parameters using gradient
            Flux.update!(optimizer, ps, gs)

            rbm_black = unwhiten(rbm_white, whiten_L, whiten_μ)

            push!(history, :epoch, epoch)
            push!(history, :batch, b)
        end

        lpl = batch_mean(log_pseudolikelihood(rbm_black, data), wts)
        push!(history, :lpl, lpl)
        if verbose
            Δt_ = round(Δt, digits=2)
            lpl_ = round(lpl, digits=2)
            println("epoch $epoch/$epochs ($(Δt_)s), log(pseudolikelihood)=$lpl_")
        end
    end

    rbm.w .= rbm_black.w
    rbm.visible.θ .= rbm_black.visible.θ
    rbm.hidden.θ .= rbm_black.hidden.θ

    return history
end

function whiten(rbm::RBM{<:Binary, <:Binary}, L::LowerTriangular, μ::AbstractVector)
    # TODO: extend to other layers
    wmat = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    g = vec(rbm.visible.θ)
    θ = vec(rbm.hidden.θ)
    return RBM(
        Binary(reshape(L \ g, size(rbm.visible)...)),
        Binary(reshape(θ + wmat' * μ, size(rbm.hidden)...)),
        reshape(L \ wmat, size(rbm.w)...)
    )
end

function unwhiten(rbm::RBM{<:Binary, <:Binary}, L::LowerTriangular, μ::AbstractVector)
    wmat = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    g = vec(rbm.visible.θ)
    θ = vec(rbm.hidden.θ)
    W = L * wmat
    return RBM(
        Binary(reshape(L * g, size(rbm.visible)...)),
        Binary(reshape(θ - W' * μ, size(rbm.hidden)...)),
        reshape(W, size(rbm.w)...)
    )
end

function whiten(v::AbstractArray, L::LowerTriangular, μ::AbstractVector)
    v_mat = reshape(v, :, size(v)[end])
    v_mat_white = L \ (v_mat .- μ)
    v_white = reshape(v_mat_white, size(v)...)
    return v_white
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
