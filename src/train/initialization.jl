"""
    initialize!(rbm, [data]; ϵ = 1e-6)

Initializes the RBM and returns it.
If provided, matches average visible unit activities from `data`.
A `StandardizedRBM` also gets its offsets and scales set from `data`
(zero offsets and unit scales without `data`).

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

function initialize!(
        rbm::RBM, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractVector{<:Real} = uniform_wts(rbm.visible, data)
    )
    @assert 0 < ϵ < 1 / 2
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert length(wts) == size(data, ndims(data)) > 0
    validate_wts(wts)
    initialize!(rbm.visible, data; ϵ, wts)
    initialize!(rbm.hidden)
    initialize_w!(rbm, data; ϵ, wts)
    zerosum!(rbm)
    return rbm
end

# fields matching mean activities `μ`, clamped away from the boundary by `ϵ`
_θ_from_mean(::Binary, μ, ϵ) = logit.(clamp.(μ, ϵ, 1 - ϵ))
_θ_from_mean(::Spin, μ, ϵ) = atanh.(clamp.(μ, ϵ - 1, 1 - ϵ))
_θ_from_mean(::_PottsLayers, μ, ϵ) = log.(clamp.(μ, ϵ, 1 - ϵ)) # not in zerosum gauge

function initialize!(
        layer::_FieldLayers, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray{<:Real} = uniform_wts(layer, data)
    )
    @assert 0 < ϵ < 1 / 2
    validate_wts(wts)
    layer.θ .= _θ_from_mean(layer, batchmean(layer, data; wts), ϵ)
    return layer
end

# Gaussian moment-matching of `θ` and `γ`, shared by the layers initialized as Gaussians.
function _initialize_gaussian_moments!(θ::AbstractArray, γ::AbstractArray, layer::AbstractLayer, data::AbstractArray; ϵ::Real, wts::AbstractArray{<:Real})
    @assert 0 < ϵ < 1 / 2
    validate_wts(wts)
    μ = batchmean(layer, data; wts)
    ν = batchvar(layer, data; wts, mean = μ)
    γ .= inv.(ν .+ ϵ)
    θ .= μ .* γ
    return layer
end

function initialize!(
        layer::Gaussian, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray{<:Real} = uniform_wts(layer, data)
    )
    return _initialize_gaussian_moments!(layer.θ, layer.γ, layer, data; ϵ, wts)
end

function initialize!(
        layer::xReLU, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray{<:Real} = uniform_wts(layer, data)
    )
    _initialize_gaussian_moments!(layer.θ, layer.γ, layer, data; ϵ, wts)
    layer.Δ .= layer.ξ .= 0
    return layer
end

function initialize!(
        layer::pReLU, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray{<:Real} = uniform_wts(layer, data)
    )
    _initialize_gaussian_moments!(layer.θ, layer.γ, layer, data; ϵ, wts)
    layer.Δ .= layer.η .= 0
    return layer
end

function initialize!(
        layer::dReLU, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray{<:Real} = uniform_wts(layer, data)
    )
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

function initialize!(
        layer::nsReLU, data::AbstractArray;
        wts::AbstractArray{<:Real} = uniform_wts(layer, data)
    )
    validate_wts(wts)
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
        λ::Real = 0.1, ϵ::Real = 1.0e-6, wts::AbstractVector{<:Real} = uniform_wts(rbm.visible, data)
    )
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert length(wts) == size(data)[end]
    validate_wts(wts)
    x = reshape(data, length(rbm.visible), size(data)[end])
    d = dot(x .* reshape(wts, 1, :), x / sum(wts))
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
