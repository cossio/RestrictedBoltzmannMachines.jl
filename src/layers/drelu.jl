struct dReLU{N,Aθp,Aθn,Aγp,Aγn} <: AbstractLayer{N}
    θp::Aθp
    θn::Aθn
    γp::Aγp
    γn::Aγn
    function dReLU(θp::AbstractArray, θn::AbstractArray, γp::AbstractArray, γn::AbstractArray)
        @assert size(θp) == size(θn) == size(γp) == size(γn)
        return new{ndims(θp), typeof(θp), typeof(θn), typeof(γp), typeof(γn)}(θp, θn, γp, γn)
    end
end

function dReLU(::Type{T}, n::Int...) where {T}
    θp = zeros(T, n...)
    θn = zeros(T, n...)
    γp = ones(T, n...)
    γn = ones(T, n...)
    return dReLU(θp, θn, γp, γn)
end

dReLU(n::Int...) = dReLU(Float64, n...)

Base.size(layer::dReLU) = size(layer.θp)
Base.size(layer::dReLU, d::Int) = size(layer.θp, d)
Base.length(layer::dReLU) = length(layer.θp)
Base.repeat(l::dReLU, n::Int...) = dReLU(
    repeat(l.θp, n...), repeat(l.θn, n...), repeat(l.γp, n...), repeat(l.γn, n...)
)

function effective(layer::dReLU, inputs::AbstractArray)
    θp = layer.θp .+ inputs
    θn = layer.θn .+ inputs
    γp = broadlike(layer.γp, θp)
    γn = broadlike(layer.γn, θn)
    return dReLU(θp, θn, γp, γn)
end

function energies(layer::dReLU, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return drelu_energy.(layer.θp, layer.θn, layer.γp, layer.γn, x)
end

free_energies(layer::dReLU) = drelu_free.(layer.θp, layer.θn, layer.γp, layer.γn)
transfer_sample(layer::dReLU) = drelu_rand.(layer.θp, layer.θn, layer.γp, layer.γn)
transfer_mode(layer::dReLU) = drelu_mode.(layer.θp, layer.θn, layer.γp, layer.γn)
transfer_std(layer::dReLU) = sqrt.(transfer_var(layer))

function transfer_mean(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)
    Fp = free_energies(lp)
    Fn = free_energies(ln)
    F = -logaddexp.(-Fp, -Fn)
    pp = exp.(F - Fp)
    pn = exp.(F - Fn)
    μp = transfer_mean(lp)
    μn = transfer_mean(ln)
    return pp .* μp - pn .* μn
end

function transfer_var(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)
    Fp = free_energies(lp)
    Fn = free_energies(ln)
    F = -logaddexp.(-Fp, -Fn)
    pp = exp.(F - Fp)
    pn = exp.(F - Fn)
    μp = transfer_mean(lp)
    μn = transfer_mean(ln)
    νp = transfer_var(lp)
    νn = transfer_var(ln)
    μ = pp .* μp - pn .* μn
    return @. pp * (νp + μp^2) + pn * (νn + μn^2) - μ^2
end

function transfer_meanvar(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)
    Fp = free_energies(lp)
    Fn = free_energies(ln)
    F = -logaddexp.(-Fp, -Fn)
    pp = exp.(F - Fp)
    pn = exp.(F - Fn)
    μp = transfer_mean(lp)
    μn = transfer_mean(ln)
    νp = transfer_var(lp)
    νn = transfer_var(ln)
    μ = pp .* μp - pn .* μn
    ν = @. pp * (νp + μp^2) + pn * (νn + μn^2) - μ^2
    return μ, ν
end

function transfer_mean_abs(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)

    Fp, Fn = free_energies(lp), free_energies(ln)
    F = -logaddexp.(-Fp, -Fn)
    pp, pn = exp.(F - Fp), exp.(F - Fn)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    return pp .* μp - pn .* μn
end

function ∂free_energy(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)
    Fp = free_energies(lp)
    Fn = free_energies(ln)
    F = -logaddexp.(-Fp, -Fn)
    pp = exp.(F - Fp)
    pn = exp.(F - Fn)
    μp, νp = transfer_meanvar(lp)
    μn, νn = transfer_meanvar(ln)
    μ2p = @. (νp + μp^2) / 2
    μ2n = @. (νn + μn^2) / 2
    return (θp = -pp .* μp, θn = pn .* μn, γp = pp .* μ2p, γn = pn .* μ2n)
end

struct dReLUStats{A}
    xp1::A; xn1::A; xp2::A; xn2::A
    function dReLUStats(layer::dReLU, data::AbstractArray; wts=nothing)
        xp = max.(data, 0)
        xn = min.(data, 0)
        xp1 = batchmean(layer, xp; wts)
        xn1 = batchmean(layer, xn; wts)
        xp2 = batchmean(layer, xp.^2; wts)
        xn2 = batchmean(layer, xn.^2; wts)
        return new{typeof(xp1)}(xp1, xn1, xp2, xn2)
    end
end

Base.size(stats::dReLUStats) = size(stats.xp1)
suffstats(layer::dReLU, data::AbstractArray; wts = nothing) = dReLUStats(layer, data; wts)

function ∂energy(layer::dReLU, stats::dReLUStats)
    @assert size(layer) == size(stats)
    return (θp = -stats.xp1, θn = -stats.xn1, γp = stats.xp2 / 2, γn = stats.xn2 / 2)
end

function drelu_energy(θp::Real, θn::Real, γp::Real, γn::Real, x::Real)
    return drelu_energy(promote(θp, θn)..., promote(γp, γn)..., x)
end

function drelu_energy(θp::T, θn::T, γp::S, γn::S, x::Real) where {T<:Real, S<:Real}
    if x ≥ 0
        return gauss_energy(θp, γp, x)
    else
        return gauss_energy(θn, γn, x)
    end
end

function drelu_free(θp::Real, θn::Real, γp::Real, γn::Real)
    Fp = relu_free( θp, γp)
    Fn = relu_free(-θn, γn)
    return -logaddexp(-Fp, -Fn)
end

function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    return drelu_rand(promote(θp, θn)..., promote(γp, γn)...)
end

function drelu_rand(θp::T, θn::T, γp::S, γn::S) where {T<:Real, S<:Real}
    Fp = relu_free(θp, γp)
    Fn = relu_free(-θn, γn)
    F = -logaddexp(-Fp, -Fn)
    if randexp(typeof(F)) ≥ Fp - F
        return  relu_rand( θp, γp)
    else
        return -relu_rand(-θn, γn)
    end
end

function drelu_mode(θp::Real, θn::Real, γp::Real, γn::Real)
    T = promote_type(typeof(θp / abs(γp)), typeof(θn / abs(γn)))
    if θp ≤ 0 ≤ θn
        return zero(T)
    elseif θn ≤ 0 ≤ θp && θp^2 / abs(γp) ≥ θn^2 / abs(γn) || θp ≥ 0 && θn ≥ 0
        return convert(T, θp / abs(γp))
    elseif θn ≤ 0 ≤ θp && θp^2 / abs(γp) ≤ θn^2 / abs(γn) || θp ≤ 0 && θn ≤ 0
        return convert(T, θn / abs(γn))
    else
        return convert(T, NaN)
    end
end
