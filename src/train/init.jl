"""
    init!(rbm, data; ϵ = 1e-6)

Inits the RBM, computing average visible unit activities from `data`.
"""
function init!(rbm::RBM, data::AbstractArray; w::Real = true, ϵ = 1e-6)
    init!(rbm.visible, data; ϵ = ϵ)
    init!(rbm.hidden)
    init_weights!(rbm; w = w)
    return rbm
end

function init!(rbm::RBM; w::Real = 1)
    init!(rbm.visible)
    init!(rbm.hidden)
    init_weights!(rbm; w=w)
    return rbm
end

function init!(layer::Potts, data::AbstractArray; ϵ::Real = 1e-6)
    @assert size(data) == (size(layer)..., size(data)[end])
    @assert 0 ≤ ϵ < 1/2
    μ = mean_(data; dims=ndims(data))
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    layer.θ .= log.(μϵ)
    layer.θ .-= mean(layer.θ; dims=1) # zerosum
    return layer
end

function init!(layer::Binary, data::AbstractArray; ϵ::Real = 1e-6)
    @assert size(data) == (size(layer)..., size(data)[end])
    @assert 0 ≤ ϵ < 1/2
    μ = mean_(data; dims = ndims(data))
    μϵ = clamp.(μ, ϵ, 1 - ϵ)
    @. layer.θ = -log(1/μϵ - 1)
    return layer
end

function init!(layer::Spin, data::AbstractArray; ϵ::Real = 1e-6)
    @assert size(data) == (size(layer)..., size(data)[end])
    @assert 0 ≤ ϵ < 1/2
    μ = mean_(data; dims = ndims(data))
    μϵ = clamp.(μ, ϵ - 1, 1 - ϵ)
    layer.θ .= atanh.(μϵ)
    return layer
end

function init!(layer::Union{Gaussian, ReLU})
    layer.θ .= 0
    layer.γ .= 1
    return layer
end

function init!(layer::dReLU)
    layer.γp .= layer.γn .= 1
    # break θp = θn = 0 symmetry
    randn!(layer.θp)
    randn!(layer.θn)
    layer.θp .*= 1e-6
    layer.θn .*= 1e-6
    return layer
end

function init!(layer::Union{Potts, Binary, Spin})
    layer.θ .= 0
    return layer
end

"""
    init_weights!(rbm; w=1)

Random initialization of weights, as independent normals with variance 1/N.
All patterns are of norm 1.
"""
function init_weights!(rbm::RBM; w::Real = 1)
    randn!(rbm.weights)
    if rbm.visible isa Potts # zerosum
        rbm.weights .-= mean(rbm.weights; dims=1)
    end
    rbm.weights .*= w / √length(rbm.visible)
    #rescale!(rbm.weights; dims=vdims(rbm))
    return rbm
end
