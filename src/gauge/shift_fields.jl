# Functions to add a constant to the fields of a layer. This is useful in many situations,
# such as zerosum, CenteredRBM, and so on.

# these must remain mutation-free, since CenteredRBM/StandardizedRBM tests differentiate
# through shift_fields with Zygote
shift_fields(l::Binary, a::AbstractArray) = Binary(; θ = l.θ .+ a)
shift_fields(l::Spin, a::AbstractArray) = Spin(; θ = l.θ .+ a)
shift_fields(l::Potts, a::AbstractArray) = Potts(; θ = l.θ .+ a)
shift_fields(l::PottsGumbel, a::AbstractArray) = PottsGumbel(; θ = l.θ .+ a)
shift_fields(l::Gaussian, a::AbstractArray) = Gaussian(; θ = l.θ .+ a, l.γ)
shift_fields(l::ReLU, a::AbstractArray) = ReLU(; θ = l.θ .+ a, l.γ)
shift_fields(l::dReLU, a::AbstractArray) = dReLU(; θp = l.θp .+ a, θn = l.θn .+ a, l.γp, l.γn)
shift_fields(l::pReLU, a::AbstractArray) = pReLU(; θ = l.θ .+ a, l.γ, l.Δ, l.η)
function shift_fields(l::xReLU{N,A,FixGamma}, a::AbstractArray) where {N,A,FixGamma}
    if FixGamma
        return xReLU(; θ = l.θ .+ a, l.Δ, l.ξ)
    else
        return xReLU(; θ = l.θ .+ a, l.γ, l.Δ, l.ξ)
    end
end

function shift_fields!(l::Union{Binary,Spin,Potts,PottsGumbel,Gaussian,ReLU,pReLU,xReLU}, a::AbstractArray)
    l.θ .+= a
    return l
end

function shift_fields!(l::dReLU, a::AbstractArray)
    l.θp .+= a
    l.θn .+= a
    return l
end
