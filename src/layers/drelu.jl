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
sample(layer::dReLU) = drelu_rand.(layer.θp, layer.θn, layer.γp, layer.γn)
mode(layer::dReLU) = drelu_mode.(layer.θp, layer.θn, layer.γp, layer.γn)

function transform_layer(layer::dReLU, inputs, β::Real = 1)
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
    θp_, θn_ = promote(θp, θn)
    γp_, γn_ = promote(γp, γn)
    if x ≥ 0
        return gauss_energy(θp_, γp_, x)
    else
        return gauss_energy(θn_, γn_, x)
    end
end

function drelu_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp = relu_cgf(+θp, γp)
    Γn = relu_cgf(-θn, γn)
    return LogExpFunctions.logaddexp(Γp, Γn)
end

function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
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
    T = promote_type(Bool, typeof(θp / abs(γp)), typeof(θn / abs(γn)))
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
