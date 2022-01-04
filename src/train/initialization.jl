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

function initialize!(layer::Binary, data::AbstractArray; ϵ::Real = 1e-6)
    @assert size(data) == (size(layer)..., size(data)[end])
    @assert 0 < ϵ < 1/2
    μ = mean_(data; dims = ndims(data))
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    @. layer.θ = -log(1/μϵ - 1)
    return layer
end

function initialize!(layer::Spin, data::AbstractArray; ϵ::Real = 1e-6)
    @assert size(data) == (size(layer)..., size(data)[end])
    @assert 0 < ϵ < 1/2
    μ = mean_(data; dims = ndims(data))
    μϵ = clamp.(μ, ϵ - 1, 1 - ϵ)
    layer.θ .= atanh.(μϵ)
    return layer
end

function initialize!(layer::Potts, data::AbstractArray; ϵ::Real = 1e-6)
    @assert size(data) == (size(layer)..., size(data)[end])
    @assert 0 < ϵ < 1/2
    μ = mean_(data; dims=ndims(data))
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    layer.θ .= log.(μϵ)
    return layer # does not do zerosum!
end

function initialize!(layer::Gaussian, data::AbstractArray; ϵ::Real = 1e-6)
    @assert size(data) == (size(layer)..., size(data)[end])
    @assert 0 < ϵ < 1/2
    μ = mean_(data; dims=ndims(data))
    ν = var_(data; dims=ndims(data))
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

"""
    initialize_weights!(rbm, data; λ = 0.1)

Initializes RBM weights such that typical inputs to hidden units are λ.
"""
function initialize_weights!(rbm::RBM, data::AbstractArray; λ::Real = 0.1)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    d = dot(data, data) / size(data)[end]
    randn!(rbm.weights)
    rbm.weights .*= λ / √d
    return rbm # does not impose zerosum
end
