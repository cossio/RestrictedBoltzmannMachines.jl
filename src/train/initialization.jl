"""
    initialize!(rbm, [data]; ϵ = 1e-6)

Initializes the RBM parameters.
If provided, matches average visible unit activities from `data`.

    initialize!(layer, [data]; ϵ = 1e-6)

Initializes a layer.
If provided, matches average unit activities from `data`.
"""
function initialize! end

function initialize!(rbm::RBM, data::AbstractArray; ϵ::Real = 1e-6)
    @assert 0 < ϵ < 1/2
    if isnothing(data)
        initialize!(rbm.visible)
    else
        @assert size(data) == (size(rbm.visible)..., size(data)[end])
        initialize!(rbm.visible, data; ϵ = ϵ)
    end
    initialize!(rbm.hidden)
    initialize_weights!(rbm, data)
    zerosum!(rbm)
    return rbm
end

function initialize!(layer::Binary, data::AbstractArray; ϵ::Real=1e-6, wts=nothing)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1/2
    μ = batchmean(layer, data; wts)
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    @. layer.θ = -log(1/μϵ - 1)
    return layer
end

function initialize!(layer::Spin, data::AbstractArray; ϵ::Real=1e-6, wts=nothing)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1/2
    μ = batchmean(layer, data; wts)
    μϵ = clamp.(μ, ϵ - 1, 1 - ϵ)
    layer.θ .= atanh.(μϵ)
    return layer
end

function initialize!(layer::Potts, data::AbstractArray; ϵ::Real=1e-6, wts=nothing)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1/2
    μ = batchmean(layer, data; wts)
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    layer.θ .= log.(μϵ)
    return layer # does not do zerosum!
end

function initialize!(layer::Gaussian, data::AbstractArray; ϵ::Real=1e-6, wts=nothing)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1/2
    μ = batchmean(layer, data; wts)
    ν = batchmean(layer, (data .- μ).^2; wts)
    layer.γ .= inv.(ν .+ ϵ)
    layer.θ .= μ .* layer.γ
    return layer
end

function initialize!(layer::Union{Potts, Binary, Spin})
    layer.θ .= 0
    return layer
end

function initialize!(layer::Union{Gaussian, ReLU})
    layer.θ .= 0
    layer.γ .= 1
    return layer
end

function initialize!(layer::dReLU)
    layer.θp .= layer.θn .= 0
    layer.γp .= layer.γn .= 1
    return layer
end

function initialize!(layer::pReLU)
    layer.θ .= layer.Δ .= layer.η .= 0
    layer.γ .= 1
    return layer
end

function initialize!(layer::xReLU)
    layer.θ .= layer.Δ .= layer.ξ .= 0
    layer.γ .= 1
    return layer
end

"""
    initialize_weights!(rbm, data; λ = 0.1)

Initializes RBM weights such that typical inputs to hidden units are λ.
"""
function initialize_weights!(
    rbm::RBM, data::AbstractTensor{N};
    λ::Real = 0.1, ϵ::Real = 1e-6
) where {N}
    @assert size(data) == (size(rbm.visible)..., size(data, N))
    d = dot(data, data) / size(data, ndims(rbm.visible) + 1)
    randn!(rbm.w)
    rbm.w .*= λ / √(d + ϵ)
    return rbm # does not impose zerosum
end

function initialize_weights!(rbm::RBM; λ::Real = 0.1, ϵ::Real = 1e-6)
    d = sum(transfer_var(rbm.visible) .+ transfer_mean(rbm.visible).^2)
    randn!(rbm.w)
    rbm.w .*= λ / √(d + ϵ)
    return rbm
end
