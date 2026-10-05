# Functions to add a constant to the fields of a layer. This is useful in many situations,
# such as zerosum, CenteredRBM, and so on.

shift_fields(l::Binary, a::AbstractArray) = Binary(; θ = l.θ .+ a)
shift_fields(l::Spin, a::AbstractArray) = Spin(; θ = l.θ .+ a)
shift_fields(l::Potts, a::AbstractArray) = Potts(; θ = l.θ .+ a)
shift_fields(l::PottsGumbel, a::AbstractArray) = PottsGumbel(; θ = l.θ .+ a)
shift_fields(l::Gaussian, a::AbstractArray) = Gaussian(; θ = l.θ .+ a, l.γ)
shift_fields(l::ReLU, a::AbstractArray) = ReLU(; θ = l.θ .+ a, l.γ)
shift_fields(l::dReLU, a::AbstractArray) = dReLU(; θp = l.θp .+ a, θn = l.θn .+ a, l.γp, l.γn)
shift_fields(l::pReLU, a::AbstractArray) = pReLU(; θ = l.θ .+ a, l.γ, l.Δ, l.η)
shift_fields(l::xReLU, a::AbstractArray) = xReLU(; θ = l.θ .+ a, l.γ, l.Δ, l.ξ)
shift_fields(l::nsReLU, a::AbstractArray) = nsReLU(; θ = l.θ .+ a, l.Δ, l.ξ)

function shift_fields!(l::_ThetaLayers, a::AbstractArray)
    l.θ .+= a
    return l
end

function shift_fields!(l::dReLU, a::AbstractArray)
    l.θp .+= a
    l.θn .+= a
    return l
end

# Gradient with respect to the shift `a` of `shift_fields(layer, a)`, pulled back from a
# gradient `∂par` with respect to the parameters of the shifted layer: the sum of the rows
# of `∂par` of the fields that the shift moves.
function ∂shift_fields(layer::_ThetaLayers, ∂par::AbstractArray)
    @assert size(∂par) == size(layer.par)
    return ∂par[1, ..]
end

function ∂shift_fields(layer::dReLU, ∂par::AbstractArray)
    @assert size(∂par) == size(layer.par)
    return ∂par[1, ..] .+ ∂par[2, ..]
end
