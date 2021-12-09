using Zygote: Grads

export gauge, zerosum, rescale, gauge!, zerosum!, rescale!

"""
    gauge(rbm)

Fixes a gauge on `rbm`, removing parameter degeneracies.
The resulting RBM share parameter vectors with the original.
"""
function gauge(rbm::RBM)::typeof(rbm)
    rbm_zs = zerosum(rbm)
    rbm_re = rescale(rbm_zs)
    return rbm_re
end

"""
    zerosum(rbm)

Set fields to zero-sum gauge. Only affects RBMs with visible or hidden
Potts layers. For other kinds of layers this function does nothing. The returned
RBM shares other parameters (e.g. hidden layer) with the original.
"""
zerosum(rbm::RBM{<:AbstractHomogeneousLayer}) = rbm
function zerosum(rbm::RBM{<:Potts})::typeof(rbm)
    vis = zerosum(rbm.visible)
    wsz = zerosum(rbm.weights; dims=1)
    return RBM(vis, rbm.hidden, wsz)
end
function zerosum(rbm::RBM{<:Potts})::typeof(rbm)
    vis = zerosum(rbm.visible)
    wsz = zerosum(rbm.weights; dims=1)
    return RBM(vis, rbm.hidden, wsz)
end
zerosum(layer::AbstractHomogeneousLayer) = layer
zerosum(layer::Potts) = Potts(zerosum(layer.θ; dims=1))
zerosum(A::AbstractArray; dims=1) = A .- mean(A; dims=dims)

"""
    rescale(rbm)

Rescales weights so that norm(w[:,μ]) = 1 for all μ (making individual weights ~ 1/√N).
The resulting RBM shares layer parameters with the original, but weights are a new array.
"""
function rescale(rbm::RBM)::typeof(rbm)
    ω = sqrt.(sum(rbm.weights.^2; dims = vdims(rbm)))
    @assert all(x -> x > 0, ω)
    weights = rbm.weights ./ ω
    return RBM(rbm.visible, rbm.hidden, weights)
end

#= In-place versions =#

function gauge!(rbm::RBM)
    zerosum!(rbm)
    rescale!(rbm)
    return rbm
end

zerosum!(rbm::RBM{<:AbstractHomogeneousLayer}) = rbm
function zerosum!(rbm::RBM{<:Potts})
    zerosum!(rbm.visible)
    zerosum!(rbm.weights; dims=1)
    return rbm
end
zerosum!(layer::AbstractHomogeneousLayer) = layer
function zerosum!(layer::Potts)
    zerosum!(layer.θ; dims=1)
    return layer
end
function zerosum!(A::AbstractArray; dims=1)
    A .-= mean(A; dims=dims)
    return A
end

function rescale!(rbm::RBM)
    rescale!(rbm.weights; dims=vdims(rbm))
    return rbm
end

function rescale!(A::AbstractArray; dims)
    λ = sqrt.(sum(A.^2; dims=dims))
    @assert all(x -> x > 0, λ)
    A ./= λ
    return A
end

"""
    gauge!(gs, rbm)

Removes gradient components perpendicular to gauge constraints.
Assumes that `rbm` is in the correct gauge.
"""
function gauge!(gs::Grads, rbm::RBM)
    zerosum!(gs, rbm)
    rescale!(gs, rbm)
    return gs
end

"""
    zerosum!(gs, rbm)

Removes gradient components perpendicular to the zerosum constraint.
Assumes that `rbm` is in the correct gauge.
"""
function zerosum!(gs::Grads, rbm::RBM{<:Potts})
    if haskey(gs, rbm.weights)# && !isnothing(gs[rbm.weights])
        gs[rbm.weights] .-= mean(gs[rbm.weights]; dims=1)
    end
    if haskey(gs, rbm.visible.θ)# && !isnothing(gs[rbm.visible.θ])
        gs[rbm.visible.θ] .-= mean(gs[rbm.visible.θ]; dims=1)
    end
    return gs
end
zerosum!(gs::Grads, rbm::RBM{<:AbstractHomogeneousLayer}) = gs

"""
    rescale!(gs, rbm)

Removes gradient components perpendicular to the weight scale constraint.
Assumes that `rbm` is in the correct gauge.
"""
function rescale!(gs::Grads, rbm::RBM)
    if haskey(gs, rbm.weights)# && !isnothing(gs[rbm.weights])
        # since RBM is in correct gauge, patterns are normalized
        λ = sum(rbm.weights .* gs[rbm.weights]; dims=vdims(rbm))
        gs[rbm.weights] .-= rbm.weights .* λ
    end
    return gs
end
#
# """
#     center!(rbm)
#
# Removes weight projections along the single-site biases.
# """
# function center!(rbm::RBM{<:Any,<:Gaussian})
#     θ = vec(rbm.visible.θ)
#     w = reshape(rbm.weights, length(θ), :)
#     w .-= θ * θ' * w ./ dot(θ, θ)
#     return rbm
# end
#
# function center(rbm::RBM{<:Any,<:Gaussian})
#     θ = vec(rbm.visible.θ)
#     w = reshape(rbm.weights, length(θ), :)
#     w -= θ * θ' * w ./ dot(θ, θ)
#     return RBM(rbm.visible, rbm.hidden, reshape(w, size(rbm.weights)))
# end
#
# center!(rbm::RBM) = rbm
# center(rbm::RBM) = rbm
#
# """
#     center_grad!(gs, rbm)
#
# Removes gradients in directions affecting the the center gauge.
# """
# function center!(gs::Grads, rbm::RBM{V,H}) where {V<:Union{Potts,Binary,Spin},
#                                                   H<:Union{Gaussian,ReLU,dReLU}}
#     if haskey(gs, rbm.visible.θ) && haskey(gs, rbm.weights)
#         g = vec(rbm.visible.θ)
#         w = reshape(rbm.weights, length(g), :)
#         A = [w' kron(I(length(rbm.hidden)), g')]
#         dw = reshape(gs[rbm.weights], length(g), :)
#         dg = vec(gs[rbm.visible.θ])
#         dx = [dg; vec(dw)]
#         y = (A * A') \ (A * dx)
#         dx .-= A' * y
#         dg .= dx[1:length(dg)]
#         vec(gs[rbm.weights]) .= dx[length(dg) + 1:end]
#     elseif haskey(gs, rbm.visible.θ)
#         dθ = vec(gs[rbm.visible.θ])
#         w = reshape(rbm.weights, length(dθ), :)
#         x = (w' * w) \ (w' * dθ)
#         dθ .-= w * x
#     elseif haskey(gs, rbm.weights)
#         θ = vec(rbm.visible.θ)
#         dw = reshape(gs[rbm.weights], length(θ), :)
#         dw .-= θ * θ' * dw ./ dot(θ, θ)
#     end
#     return rbm
# end
#
# center!(gs::Grads, rbm::RBM) = rbm