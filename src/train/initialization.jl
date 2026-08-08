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

#= `initialize!` boundaries default to lazy uniform weights and apply the
same hygiene as training to the weights: validation and dropping zero-weight
samples (so non-finite data attached to them cannot poison the
moment-matching reductions). Returns the prepared `(data, wts)`. =#
function _initialization_data(wts::AbstractArray, layer::AbstractLayer, data::AbstractArray)
    @assert size(wts) == batch_size(layer, data)
    wts = _validate_weights(wts)
    # Flatten the trailing batch dimensions, so zero-weight configurations are
    # dropped uniformly regardless of the batch layout; the moment-matching
    # reductions are invariant under this reshape.
    return _filter_zero_weights(reshape(data, size(layer)..., :), vec(wts))
end
function _initialization_data(wts::AbstractVector, data::AbstractArray)
    length(wts) == size(data, ndims(data)) ||
        throw(DimensionMismatch("length(wts) must equal the number of data samples"))
    wts = _validate_weights(wts)
    return _filter_zero_weights(data, wts)
end

function initialize!(
        rbm::RBM, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractVector = Ones{Bool}(size(data, ndims(data)))
    )
    @assert 0 < ϵ < 1 / 2
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    data, wts = _initialization_data(wts, data)
    initialize!(rbm.visible, data; ϵ, wts)
    initialize!(rbm.hidden)
    initialize_w!(rbm, data; ϵ, wts)
    zerosum!(rbm)
    return rbm
end

function initialize!(
        layer::Binary, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray = uniform_weights(layer, data)
    )
    data, wts = _initialization_data(wts, layer, data)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1 / 2
    μ = batchmean(layer, data; wts)
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    layer.θ .= logit.(μϵ)
    return layer
end

function initialize!(
        layer::Spin, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray = uniform_weights(layer, data)
    )
    data, wts = _initialization_data(wts, layer, data)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1 / 2
    μ = batchmean(layer, data; wts)
    μϵ = clamp.(μ, ϵ - 1, 1 - ϵ)
    layer.θ .= atanh.(μϵ)
    return layer
end

function initialize!(
        layer::Union{Potts, PottsGumbel}, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray = uniform_weights(layer, data)
    )
    data, wts = _initialization_data(wts, layer, data)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1 / 2
    μ = batchmean(layer, data; wts)
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    layer.θ .= log.(μϵ)
    return layer # does not do zerosum!
end

# Gaussian moment-matching of `θ` and `γ`, shared by the layers initialized as Gaussians.
function _initialize_gaussian_moments!(θ::AbstractArray, γ::AbstractArray, layer::AbstractLayer, data::AbstractArray; ϵ::Real, wts::AbstractArray)
    @assert size(layer) == size(data)[1:ndims(layer)]
    @assert 0 < ϵ < 1 / 2
    μ = batchmean(layer, data; wts)
    ν = batchmean(layer, (data .- μ) .^ 2; wts)
    γ .= inv.(ν .+ ϵ)
    θ .= μ .* γ
    return layer
end

function initialize!(
        layer::Gaussian, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray = uniform_weights(layer, data)
    )
    data, wts = _initialization_data(wts, layer, data)
    return _initialize_gaussian_moments!(layer.θ, layer.γ, layer, data; ϵ, wts)
end

function initialize!(
        layer::xReLU, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray = uniform_weights(layer, data)
    )
    data, wts = _initialization_data(wts, layer, data)
    _initialize_gaussian_moments!(layer.θ, layer.γ, layer, data; ϵ, wts)
    layer.Δ .= layer.ξ .= 0
    return layer
end

function initialize!(
        layer::pReLU, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray = uniform_weights(layer, data)
    )
    data, wts = _initialization_data(wts, layer, data)
    _initialize_gaussian_moments!(layer.θ, layer.γ, layer, data; ϵ, wts)
    layer.Δ .= layer.η .= 0
    return layer
end

function initialize!(
        layer::dReLU, data::AbstractArray;
        ϵ::Real = 1.0e-6, wts::AbstractArray = uniform_weights(layer, data)
    )
    data, wts = _initialization_data(wts, layer, data)
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
        wts::AbstractArray = uniform_weights(layer, data)
    )
    data, wts = _initialization_data(wts, layer, data)
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
        λ::Real = 0.1, ϵ::Real = 1.0e-6, wts::AbstractVector = Ones{Bool}(size(data, ndims(data)))
    )
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    data, wts = _initialization_data(wts, data)
    @assert length(wts) == size(data)[end]
    x = reshape(data, length(rbm.visible), size(data)[end])
    # dividing `x` first floats the accumulator, so narrow integer data
    # (e.g. Int8 spins) cannot overflow the dot product
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
