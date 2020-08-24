export dReLU

struct dReLU{T,N} <: AbstractLayer{T,N}
    θp::Array{T,N}
    θn::Array{T,N}
    γp::Array{T,N}
    γn::Array{T,N}
    function dReLU{T,N}(ps::Vararg{Array{T,N}, 4}) where {T,N}
        allequal(size.(ps)...) || pardimserror()
        return new{T,N}(ps...)
    end
end
dReLU(ps::Vararg{Array{T,N}, 4}) where {T,N} = dReLU{T,N}(ps...)
function dReLU(p::Union{Gaussian,ReLU}, n::Union{Gaussian,ReLU})
    size(p) == size(n) || pardimserror()
    return dReLU(p.θ, -n.θ, p.γ, n.γ)
end
dReLU{T}(n::Int...) where {T} = dReLU(ReLU{T}(n...), ReLU{T}(n...))
dReLU(n::Int...) = dReLU{Float64}(n...)
Flux.@functor dReLU
fields(l::dReLU) = (l.θp, l.θn, l.γp, l.γn)
relus_pair(l::dReLU) = ReLU(l.θp, l.γp), ReLU(-l.θn, l.γn)
gauss_pair(l::dReLU) = Gaussian(l.θp, l.γp), Gaussian(-l.θn, l.γn)

function probs_pair(layer::dReLU)
    lp, ln = relus_pair(layer)
    Γp, Γn = _cgf(lp), _cgf(ln)
    Γ = logaddexp.(Γp, Γn)
    pp = @. exp(Γp - Γ)
    pn = @. exp(Γn - Γ)
    return pp, pn
end

function __energy(layer::dReLU, x::AbstractArray)
    checkdims(layer, x)
    xp = @. max( x, zero(x))
    xn = @. max(-x, zero(x))
    lp, ln = relus_pair(layer)
    Ep, En = __energy(lp, xp), __energy(ln, xn)
    return Ep .+ En
end

_cgf(layer::dReLU) = drelu_cgf.(layer.θp, layer.θn, layer.γp, layer.γn)
_random(layer::dReLU) = drelu_rand.(layer.θp, layer.θn, layer.γp, layer.γn)

function _transfer_mode(layer::dReLU)
    lp, ln = relus_pair(layer)
    xp, xn = _transfer_mode(lp), -_transfer_mode(ln)
    Ep, En = __energy(lp, +xp), __energy(ln, -xn)
    return @. ifelse(Ep ≤ En, xp, xn)
end

function effective_β(layer::dReLU, β)
    p, n = relus_pair(layer)
    return dReLU(effective_β(p, β), effective_β(n, β))
end

function effective_I(layer::dReLU, I)
    p, n = relus_pair(layer)
    return dReLU(effective_I(p, I), effective_I(n, -I))
end

function _transfer_mean(layer::dReLU)
    lp, ln = relus_pair(layer)
    pp, pn = probs_pair(layer)
    return pp .* _transfer_mean(lp) .- pn .* _transfer_mean(ln)
end

_transfer_std(layer::dReLU) = sqrt.(_transfer_var(layer))

function _transfer_var(layer::dReLU)
    lp, ln = relus_pair(layer)
    pp, pn = probs_pair(layer)
    μp, μn = _transfer_mean(lp), _transfer_mean(ln)
    νp, νn = _transfer_var(lp),  _transfer_var(ln)
    μ = @. pp * μp - pn * μn
    return @. pp * (νp + μp^2) + pn * (νn + μn^2) - μ^2
end

function _transfer_mean_abs(layer::dReLU)
    lp, ln = relus_pair(layer)
    pp, pn = probs_pair(layer)
    return pp .* _transfer_mean(lp) .+ pn .* _transfer_mean(ln)
end

#=
Compute gradients using the approach of:
http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients

It is easier to write the adjoint for a function that takes scalar inputs,
than for a function that takes struct inputs.
=#

function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp = relu_cgf(+θp, γp)
    Γn = relu_cgf(-θn, γn)
    Γ = logaddexp(Γp, Γn)
    if rand(typeof(Γ)) ≤ exp(Γp - Γ)
        return +relu_rand(+θp, γp)
    else
        return -relu_rand(-θn, γn)
    end
end
@adjoint function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    z, dθp, dθn, dγp, dγn = ∇drelu_rand(θp, θn, γp, γn)
    return z, δ -> (δ * dθp, δ * dθn, δ * dγp, δ * dγn)
end
@adjoint function broadcasted(::typeof(drelu_rand), θp::Numeric, θn::Numeric, γp::Numeric, γn::Numeric)
    z, dθp, dθn, dγp, dγn = ∇drelu_rand(θp, θn, γp, γn)
    return z, δ -> (nothing, δ .* dθp, δ .* dθn, δ .* dγp, δ .* dγn)
end
function ∇drelu_rand(θp::Numeric, θn::Numeric, γp::Numeric, γn::Numeric)
    zipped = ∇drelu_rand.(θp, θn, γp, γn)
    return unzip(zipped)
end
function ∇drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp, dΓp_dθp, dΓp_dγp = ∇relu_cgf(+θp, γp)
    Γn, dΓn_dθn, dΓn_dγn = ∇relu_cgf(-θn, γn)
    Γ = logaddexp(Γp, Γn)
    pp = exp(Γp - Γ)
    pn = exp(Γn - Γ)
    if rand_like(Γ) ≤ pp
        xp, dxp_dθp, dxp_dγp = ∇relu_rand(+θp, γp)
        m = relu_mills(θp, γp, xp)
        dθp = m * (1 - pp) * dΓp_dθp + dxp_dθp
        dγp = m * (1 - pp) * dΓp_dγp + dxp_dγp
        dθn = -m * pn * dΓn_dθn
        dγn = -m * pn * dΓn_dγn
        return +xp, dθp, -dθn, dγp, dγn
    else
        xn, dxn_dθn, dxn_dγn = ∇relu_rand(-θn, γn)
        m = relu_mills(-θn, γn, xn)
        dθn = -m * (1 - pn) * dΓn_dθn - dxn_dθn
        dγn = -m * (1 - pn) * dΓn_dγn - dxn_dγn
        dθp = m * pp * dΓp_dθp
        dγp = m * pp * dΓp_dγp
        return -xn, dθp, -dθn, dγp, dγn
    end
end

drelu_survival(θp::Real, θn::Real, γp::Real, γn::Real, x::Real) =
    exp(drelu_logsurvival(θp, θn, γp, γn, x))
function drelu_logsurvival(θp::Real, θn::Real, γp::Real, γn::Real, x::Real)
    Γp, Γn = relu_cgf(θp, γp), relu_cgf(-θn, γn)
    Γ = logaddexp(Γp, Γn)
    if x < 0
        return log1mexp(Γn - Γ + relu_logsurvival(-θn, γn, -x))
    else
        return Γp - Γ + relu_logsurvival(+θp, γp, +x)
    end
end

drelu_pdf(θp::Real, θn::Real, γp::Real, γn::Real, x::Real) =
    exp(drelu_logpdf(θp, θn, γp, γn, x))
function drelu_logpdf(θp::Real, θn::Real, γp::Real, γn::Real, x::Real)
    Γp = relu_cgf(+θp, γp)
    Γn = relu_cgf(-θn, γn)
    Γ = logaddexp(Γp, Γn)
    if x > 0
        return -(abs(γp) * x/2 - θp) * x - Γ
    else
        return -(abs(γn) * x/2 - θn) * x - Γ
    end
end

function drelu_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp, Γn = relu_cgf(θp, γp), relu_cgf(-θn, γn)
    return logaddexp(Γp, Γn)
end
