"""
    initialize!(rbm, [data]; ϵ = 1e-6)

Initializes the RBM and returns it.
If provided, matches average visible unit activities from `data`.

    initialize!(layer, [data]; ϵ = 1e-6)

Initializes a layer and returns it.
If provided, matches average unit activities from `data`.
"""
function initialize! end

function initialize!(rbm::RBM; ϵ::Real = 1.0e-6)
    initialize!(rbm.visible)
    initialize!(rbm.hidden)
    initialize_w!(rbm; ϵ)
    zerosum!(rbm)
    return rbm
end

function initialize!(rbm::RBM, data::AbstractArray; ϵ::Real = 1.0e-6, wts = nothing)
    @assert 0 < ϵ < 1 / 2
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    initialize!(rbm.visible, data; ϵ, wts)
    initialize!(rbm.hidden)
    initialize_w!(rbm, data; ϵ, wts)
    zerosum!(rbm)
    return rbm
end

function initialize!(layer::Binary, data::AbstractArray; ϵ::Real = 1.0e-6, wts = nothing)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1 / 2
    μ = batchmean(layer, data; wts)
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    @. layer.θ = -log(1 / μϵ - 1)
    return layer
end

function initialize!(layer::Spin, data::AbstractArray; ϵ::Real = 1.0e-6, wts = nothing)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1 / 2
    μ = batchmean(layer, data; wts)
    μϵ = clamp.(μ, ϵ - 1, 1 - ϵ)
    layer.θ .= atanh.(μϵ)
    return layer
end

function initialize!(layer::Union{Potts, PottsGumbel}, data::AbstractArray; ϵ::Real = 1.0e-6, wts = nothing)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1 / 2
    μ = batchmean(layer, data; wts)
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    layer.θ .= log.(μϵ)
    return layer # does not do zerosum!
end

# Gaussian moment-matching of `θ` and `γ`, shared by the layers initialized as Gaussians.
function _initialize_gaussian_moments!(θ::AbstractArray, γ::AbstractArray, layer::AbstractLayer, data::AbstractArray; ϵ::Real, wts)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1 / 2
    μ = batchmean(layer, data; wts)
    ν = batchmean(layer, (data .- μ) .^ 2; wts)
    γ .= inv.(ν .+ ϵ)
    θ .= μ .* γ
    return layer
end

function initialize!(layer::Gaussian, data::AbstractArray; ϵ::Real = 1.0e-6, wts = nothing)
    return _initialize_gaussian_moments!(layer.θ, layer.γ, layer, data; ϵ, wts)
end

function initialize!(layer::xReLU, data::AbstractArray; ϵ::Real = 1.0e-6, wts = nothing)
    _initialize_gaussian_moments!(layer.θ, layer.γ, layer, data; ϵ, wts)
    layer.Δ .= layer.ξ .= 0
    return layer
end

function initialize!(layer::pReLU, data::AbstractArray; ϵ::Real = 1.0e-6, wts = nothing)
    _initialize_gaussian_moments!(layer.θ, layer.γ, layer, data; ϵ, wts)
    layer.Δ .= layer.η .= 0
    return layer
end

function initialize!(layer::dReLU, data::AbstractArray; ϵ::Real = 1.0e-6, wts = nothing)
    # initialize as Gaussian
    _initialize_gaussian_moments!(layer.θp, layer.γp, layer, data; ϵ, wts)
    layer.θn .= layer.θp
    layer.γn .= layer.γp
    return layer
end

function initialize!(layer::Union{Binary, Spin, Potts, PottsGumbel})
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

function initialize!(layer::nsReLU, data::AbstractArray; wts = nothing)
    @assert size(layer) == size(data)[1:ndims(layer)]
    μ = batchmean(layer, data; wts)
    layer.θ .= μ
    layer.Δ .= layer.ξ .= 0
    return layer
end

function initialize!(layer::nsReLU)
    layer.θ .= layer.Δ .= layer.ξ .= 0
    return layer
end

"""
    initialize_w!(rbm, data; λ = 0.1)

Initializes `rbm.w` such that typical inputs to hidden units are λ.
"""
function initialize_w!(
        rbm::RBM, data::AbstractArray;
        λ::Real = 0.1, ϵ::Real = 1.0e-6, wts::Union{Nothing, AbstractVector} = nothing
    )
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    if isnothing(wts)
        d = dot(data, data / size(data, ndims(data)))
    else
        @assert length(wts) == size(data)[end]
        x = reshape(data, length(rbm.visible), size(data)[end])
        d = dot(x * Diagonal(wts), x / sum(wts))
    end
    randn!(rbm.w)
    rbm.w .*= λ / √(d + ϵ)
    return rbm # does not impose zerosum
end

function initialize_w!(rbm::RBM; λ::Real = 0.1, ϵ::Real = 1.0e-6)
    d = sum(var_from_inputs(rbm.visible) .+ mean_from_inputs(rbm.visible) .^ 2)
    randn!(rbm.w)
    rbm.w .*= λ / √(d + ϵ)
    return rbm
end
