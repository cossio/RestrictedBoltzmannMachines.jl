export init!, init_weights!

"""
    init!(rbm, data; eps = 1e-6)

Inits the RBM, computing average visible unit activities from `data`.
"""
function init!(rbm::RBM, data::AbstractArray; w::Real=1, eps = 1e-6)
    init!(rbm.vis, data; eps = eps)
    init!(rbm.hid)
    init_weights!(rbm; w=w)
    return rbm
end

function init!(rbm::RBM; w::Real=1)
    init!(rbm.vis)
    init!(rbm.hid)
    init_weights!(rbm; w=w)
    return rbm
end

function init!(layer::Potts, data::AbstractArray; eps=1e-6)
    0 ≤ eps < 1/2 || throw(ArgumentError("got eps = $eps"))
    checkdims(layer, data)
    bdims = batchdims(layer, data)
    μ = meandrop(data; dims=bdims)
    clamp!(μ, eps, 1 - eps)
    layer.θ .= log.(μ)
    layer.θ .-= mean(layer.θ; dims=1) # zerosum
    return layer
end

function init!(layer::Binary, data::AbstractArray; eps=1e-6)
    0 ≤ eps < 1/2 || throw(ArgumentError("got eps = $eps"))
    checkdims(layer, data)
    bdims = batchdims(layer, data)
    μ = meandrop(data; dims=bdims)
    clamp!(μ, eps, 1 - eps)
    @. layer.θ = log(μ / (1 - μ))
    return layer
end

function init!(layer::Spin, data::AbstractArray; eps=1e-6)
    eps ≥ 0 || throw(ArgumentError("got eps = $eps"))
    checkdims(layer, data)
    bdims = batchdims(layer, data)
    μ = meandrop(data; dims=bdims)
    layer.θ .= atanh.(clamp.(μ, eps - 1, 1 - eps))
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
    if rbm.vis isa Potts # zerosum
        rbm.weights .-= mean(rbm.weights; dims=1)
    end
    rbm.weights .*= w / √length(rbm.vis)
    #rescale!(rbm.weights; dims=vdims(rbm))
    return rbm
end

function init_cov!(rbm::RBM, data::NumArray; eps = 1e-6)
    error("Not implemented")
    init!(rbm.vis, data; eps = eps)
    init!(rbm.hid)
    data_mat = reshape(data, length(rbm.vis), :)
    cov(data_mat)
    init_weights!(rbm; w=1)
    return rbm
end
