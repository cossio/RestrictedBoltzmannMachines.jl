"""
    rescale_hidden!(rbm, λ::AbstractArray)

For continuous hidden units with a scale parameter, scales parameters such that hidden
unit activations are divided by `λ`. For other hidden units does nothing. The resulting RBM
is equivalent to the original one.
"""
function rescale_hidden!(rbm::RBM, λ::AbstractArray)
    @assert size(rbm.hidden) == size(λ)
    if rescale_activations!(rbm.hidden, λ)
        rbm.w .*= reshape(λ, map(one, size(rbm.visible))..., size(rbm.hidden)...)
    end
    return rbm
end

"""
    rescale_weights!(rbm, λ::AbstractArray)

For continuous hidden units with a scale parameter, scales parameters such that the weights
attached to each hidden unit have norm 1.
"""
function rescale_weights!(rbm::RBM)
    ω = weight_norms(rbm)
    λ = inv.(ω)
    return rescale_hidden!(rbm, λ)
end

function weight_norms(rbm::RBM)
    w2 = sum(abs2, rbm.w; dims=1:ndims(rbm.visible))
    return reshape(sqrt.(w2), size(rbm.hidden))
end

"""
    rescale_activations!(layer, λ::AbstractArray)

For continuous layers with scale parameters, re-parameterizes such that unit activations
are divided by `λ`, and returns `true`. For other layers, does nothing and returns `false`.
"""
rescale_activations!(layer::Union{Binary,Spin,Potts}, λ::AbstractArray) = false

#= Note that λ < 0 can lead to trouble, e.g. for ReLU which
must have positive activations. So we dissallow it below. =#

function rescale_activations!(layer::Union{Gaussian,ReLU}, λ::AbstractArray)
    @assert size(layer) == size(λ)
    @assert all(>(0), λ)
    layer.θ .*= λ
    layer.γ .*= λ.^2
    return true
end

function rescale_activations!(layer::dReLU, λ::AbstractArray)
    @assert size(layer) == size(λ)
    @assert all(>(0), λ)
    layer.θp .*= λ
    layer.θn .*= λ
    layer.γp .*= λ.^2
    layer.γn .*= λ.^2
    return true
end

function rescale_activations!(layer::Union{pReLU,xReLU}, λ::AbstractArray)
    @assert size(layer) == size(λ)
    @assert all(>(0), λ) # it's just simpler
    layer.θ .*= λ
    layer.Δ .*= λ
    layer.γ .*= λ.^2
    return true
end
