# Functions to add a constant to the fields of a layer. This is useful in many situations,
# such as zerosum, CenteredRBM, and so on.

shift_fields(l::Binary, a::AbstractArray) = Binary(; θ = l.θ .+ a)
shift_fields(l::Spin, a::AbstractArray) = Spin(; θ = l.θ .+ a)
shift_fields(l::Potts, a::AbstractArray) = Potts(; θ = l.θ .+ a)
shift_fields(l::Gaussian, a::AbstractArray) = Gaussian(; θ = l.θ .+ a, l.γ)
shift_fields(l::ReLU, a::AbstractArray) = ReLU(; θ = l.θ .+ a, l.γ)
shift_fields(l::dReLU, a::AbstractArray) = dReLU(; θp = l.θp .+ a, θn = l.θn .+ a, l.γp, l.γn)
shift_fields(l::pReLU, a::AbstractArray) = pReLU(; θ = l.θ .+ a, l.γ, l.Δ, l.η)
shift_fields(l::xReLU, a::AbstractArray) = xReLU(; θ = l.θ .+ a, l.γ, l.Δ, l.ξ)

function shift_fields!(l::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU}, a::AbstractArray)
    l.θ .+= a
    return l
end

function shift_fields!(l::dReLU, a::AbstractArray)
    l.θp .+= a
    l.θn .+= a
    return l
end
