struct dReLU{A<:AbstractArray}
    θp::A
    θn::A
    γp::A
    γn::A
    function dReLU(θp::A, θn::A, γp::A, γn::A) where {A<:AbstractArray}
        @assert size(θp) == size(θn) == size(γp) == size(γn)
        return new{A}(θp, θn, γp, γn)
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

function energies(layer::dReLU, x::AbstractArray)
    return drelu_energy.(layer.θp, layer.θn, layer.γp, layer.γn, x)
end

cgfs(layer::dReLU) = drelu_cgf.(layer.θp, layer.θn, layer.γp, layer.γn)
transfer_sample(layer::dReLU) = drelu_rand.(layer.θp, layer.θn, layer.γp, layer.γn)
transfer_mode(layer::dReLU) = drelu_mode.(layer.θp, layer.θn, layer.γp, layer.γn)

function transfer_mean(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)

    Γp, Γn = cgfs(lp), cgfs(ln)
    Γ = LogExpFunctions.logaddexp.(Γp, Γn)
    pp, pn = exp.(Γp - Γ), exp.(Γn - Γ)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    return pp .* μp + pn .* μn
end

function transfer_var(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)

    Γp, Γn = cgfs(lp), cgfs(ln)
    Γ = LogExpFunctions.logaddexp.(Γp, Γn)
    pp, pn = exp.(Γp - Γ), exp.(Γn - Γ)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    νp, νn = transfer_var(lp), transfer_var(ln)
    μ = pp .* μp + pn .* μn

    return @. pp * (νp + μp^2) + pn * (νn + μn^2) - μ^2
end

function transfer_mean_abs(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)

    Γp, Γn = cgfs(lp), cgfs(ln)
    Γ = LogExpFunctions.logaddexp.(Γp, Γn)
    pp, pn = exp.(Γp - Γ), exp.(Γn - Γ)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    return pp .* μp - pn .* μn
end

function conjugates(layer::dReLU)
    lp = ReLU( layer.θp, layer.γp)
    ln = ReLU(-layer.θn, layer.γn)

    Γp, Γn = cgfs(lp), cgfs(ln)
    Γ = LogExpFunctions.logaddexp.(Γp, Γn)
    pp, pn = exp.(Γp - Γ), exp.(Γn - Γ)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    νp, νn = transfer_var(lp), transfer_var(ln)

    μ2p = @. νp + μp^2
    μ2n = @. νn + μn^2

    return (
        θp = pp .* μp,
        θn = pn .* μn,
        γp = -pp .* μ2p/2,
        γn = -pn .* μ2n/2
    )
end

function conjugates_empirical(layer::dReLU, samples::AbstractArray)
    @assert size(samples) == (size(layer)..., size(samples)[end])
    xp = max.(samples, 0)
    xn = min.(samples, 0)

    μp = mean_(xp; dims=ndims(samples))
    μn = mean_(xn; dims=ndims(samples))

    μ2p = mean_(xp.^2; dims=ndims(samples))
    μ2n = mean_(xn.^2; dims=ndims(samples))

    return (θp = μp, θn = μn, γp = -μ2p/2, γn = -μ2n/2)
end

function effective(layer::dReLU, inputs, β::Real = 1)
    θp = β * (layer.θp .+ inputs)
    θn = β * (layer.θn .+ inputs)
    γp = β * broadlike(layer.γp, inputs)
    γn = β * broadlike(layer.γn, inputs)
    return dReLU(promote(θp, θn, γp, γn)...)
end

Base.size(layer::dReLU) = size(layer.θp)
Base.size(layer::dReLU, d::Int) = size(layer.θp, d)
Base.ndims(layer::dReLU) = ndims(layer.θp)
Base.length(layer::dReLU) = length(layer.θp)

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

function drelu_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp = relu_cgf( θp, γp)
    Γn = relu_cgf(-θn, γn)
    return LogExpFunctions.logaddexp(Γp, Γn)
end

function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    return drelu_rand(promote(θp, θn)..., promote(γp, γn)...)
end

function drelu_rand(θp::T, θn::T, γp::S, γn::S) where {T<:Real, S<:Real}
    Γp = relu_cgf( θp, γp)
    Γn = relu_cgf(-θn, γn)
    Γ = LogExpFunctions.logaddexp(Γp, Γn)
    if randexp(typeof(Γ)) ≥ Γ - Γp
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
