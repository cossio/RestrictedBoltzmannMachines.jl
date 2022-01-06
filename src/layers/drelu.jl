struct dReLU{N, T, A <: AbstractArray{T,N}} <: AbstractLayer{N}
    θp::A
    θn::A
    γp::A
    γn::A
    function dReLU(θp::A, θn::A, γp::A, γn::A) where {A<:AbstractArray}
        @assert size(θp) == size(θn) == size(γp) == size(γn)
        return new{ndims(A), eltype(A), A}(θp, θn, γp, γn)
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

Flux.@functor dReLU

function effective(layer::dReLU, inputs::AbstractTensor; β::Real = true)
    θp = β * (layer.θp .+ inputs)
    θn = β * (layer.θn .+ inputs)
    γp = β * broadlike(layer.γp, θp)
    γn = β * broadlike(layer.γn, θn)
    return dReLU(promote(θp, θn, γp, γn)...)
end

function energies(layer::dReLU, x::AbstractTensor)
    check_size(layer, x)
    return drelu_energy.(layer.θp, layer.θn, layer.γp, layer.γn, x)
end

free_energies(layer::dReLU) = drelu_free.(layer.θp, layer.θn, layer.γp, layer.γn)
transfer_sample(layer::dReLU) = drelu_rand.(layer.θp, layer.θn, layer.γp, layer.γn)
transfer_mode(layer::dReLU) = drelu_mode.(layer.θp, layer.θn, layer.γp, layer.γn)
transfer_std(layer::dReLU) = sqrt.(transfer_var(layer))

function transfer_mean(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)

    Fp, Fn = free_energies(lp), free_energies(ln)
    F = -LogExpFunctions.logaddexp.(-Fp, -Fn)
    pp, pn = exp.(F - Fp), exp.(F - Fn)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    return pp .* μp + pn .* μn
end

function transfer_var(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)

    Fp, Fn = free_energies(lp), free_energies(ln)
    F = -LogExpFunctions.logaddexp.(-Fp, -Fn)
    pp, pn = exp.(F - Fp), exp.(F - Fn)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    νp, νn = transfer_var(lp), transfer_var(ln)
    μ = pp .* μp + pn .* μn

    return @. pp * (νp + μp^2) + pn * (νn + μn^2) - μ^2
end

function transfer_mean_abs(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)

    Fp, Fn = free_energies(lp), free_energies(ln)
    F = -LogExpFunctions.logaddexp.(-Fp, -Fn)
    pp, pn = exp.(F - Fp), exp.(F - Fn)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    return pp .* μp - pn .* μn
end

Base.size(layer::dReLU) = size(layer.θp)
Base.size(layer::dReLU, d::Int) = size(layer.θp, d)
Base.ndims(layer::dReLU) = ndims(layer.θp)
Base.length(layer::dReLU) = length(layer.θp)

function ∂free_energy(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)

    Fp, Fn = free_energies(lp), free_energies(ln)
    F = -LogExpFunctions.logaddexp.(-Fp, -Fn)
    pp, pn = exp.(F - Fp), exp.(F - Fn)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    νp, νn = transfer_var(lp), transfer_var(ln)

    μ2p = @. (νp + μp^2) / 2
    μ2n = @. (νn + μn^2) / 2

    return (θp = -pp .* μp, θn = -pn .* μn, γp = pp .* μ2p, γn = pn .* μ2n)
end

function ∂energy(layer::dReLU; xp, xn, xn2, xp2)
    for ξ in (xp, xn, xp2, xn2)
        @assert size(ξ::AbstractTensor) == size(layer)
    end
    return (θp = -xp, θn = -xn, γp = xp2 / 2, γn = xn2 / 2)
end

function sufficient_statistics(layer::dReLU, x::AbstractTensor, wts::Wts)
    check_size(layer, x)
    @assert size(x) == (size(layer)..., size(x)[end])
    xp = max.(x, 0)
    xn = min.(x, 0)
    μp = batch_mean(xp, wts)
    μn = batch_mean(xn, wts)
    μp2 = batch_mean(xp.^2, wts)
    μn2 = batch_mean(xn.^2, wts)
    return (; xp = μp, xn = μn, xp2 = μp2, xn2 = μn2)
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
    return -LogExpFunctions.logaddexp(-Fp, -Fn)
end

function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    return drelu_rand(promote(θp, θn)..., promote(γp, γn)...)
end

function drelu_rand(θp::T, θn::T, γp::S, γn::S) where {T<:Real, S<:Real}
    Fp, Fn = relu_free(θp, γp), relu_free(-θn, γn)
    F = -LogExpFunctions.logaddexp(-Fp, -Fn)
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
