"""
    rescale_hidden!(rbm, λ::AbstractArray)

For continuous hidden units with a scale parameter,
scales parameters such that hidden unit activations
are multiplied by `λ`. For other hidden units does
nothing. The resulting RBM is equivalent to the
original one.
"""
function rescale_hidden!(rbm::RBM, λ::AbstractArray)
    @assert size(hidden(rbm)) == size(λ)
    if rescale_activations!(hidden(rbm), λ)
        λ_sz = (map(Returns(1), size(visible(rbm)))..., size(hidden(rbm))...)
        weights(rbm) ./= reshape(λ, λ_sz)
    end
    return rbm
end

"""
    rescale_activations!(layer, λ::AbstractArray)

For continuous layers with scale parameters, re-parameterizes
such that unit activations are multiplied by `λ`, and returns `true`.
For other layers just returns `false`.
"""
rescale_activations!(layer::AbstractLayer, λ::AbstractArray) = false

#= Note that λ < 0 can lead to trouble, e.g. for ReLU which
must have positive activations. So we dissallow it below. =#

function rescale_activations!(layer::Union{Gaussian,ReLU}, λ::AbstractArray)
    @assert size(layer) == size(λ)
    @assert all(λ .> 0)
    layer.θ ./= λ
    layer.γ ./= λ.^2
    return true
end

function rescale_activations!(layer::dReLU, λ::AbstractArray)
    @assert size(layer) == size(λ)
    @assert all(λ .> 0)
    layer.θp ./= λ
    layer.θn ./= λ
    layer.γp ./= λ.^2
    layer.γn ./= λ.^2
    return true
end

function rescale_activations!(layer::Union{pReLU,xReLU}, λ::AbstractArray)
    @assert size(layer) == size(λ)
    @assert all(λ .> 0) # makes life simpler
    layer.θ ./= λ
    layer.Δ ./= λ
    layer.γ ./= λ.^2
    return true
end
