"""
    initialize!(rbm, [data]; ϵ = 1e-6)

Initializes the RBM parameters.
If `data` is provided, matches average visible unit activities from `data`.
"""
function initialize! end

function initialize!(rbm::RBM, data::AbstractArray; ϵ = 1e-6)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    initialize!(rbm.visible, data; ϵ = ϵ)
    initialize!(rbm.hidden)
    initialize_weights!(rbm)
    return rbm
end

function initialize!(rbm::RBM)
    initialize!(rbm.visible)
    initialize!(rbm.hidden)
    initialize_weights!(rbm)
    return rbm
end

function initialize!(layer::Potts, data::AbstractArray; ϵ::Real = 1e-6)
    @assert size(data) == (size(layer)..., size(data)[end])
    @assert 0 < ϵ < 1/2
    μ = mean_(data; dims=ndims(data))
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    layer.θ .= log.(μϵ)
    layer.θ .-= mean(layer.θ; dims=1) # zerosum
    return layer
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
    initialize_weights!(rbm)

Random initialization of weights, as independent normals with variance 1/N.
"""
function initialize_weights!(rbm::RBM)
    randn!(rbm.weights)
    if rbm.visible isa Potts # zerosum
        zerosum!(rbm.weights; dims=1)
    end
    rbm.weights .*= w / √length(rbm.visible)
    return rbm
end

function initialize_weights!(rbm::RBM)
    randn!(rbm.weights)
    if rbm.visible isa Potts # zerosum
        rbm.weights .-= mean(rbm.weights; dims=1)
    end
    rbm.weights ./= 10√length(rbm.visible)
    return rbm
end
