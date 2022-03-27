struct StdXReLU{N,Aθ,AΔ,Aξ} <: AbstractLayer{N}
    θ::Aθ
    Δ::AΔ
    ξ::Aξ
    function StdXReLU(θ::AbstractArray, Δ::AbstractArray, ξ::AbstractArray)
        @assert size(θ) == size(Δ) == size(ξ)
        return new{ndims(θ), typeof(θ), typeof(Δ), typeof(ξ)}(θ, Δ, ξ)
    end
end

"""
    StdXReLU([T], n...)

Standardized xReLU layer, with unit scale parameters.
"""
StdXReLU(::Type{T}, n::NTuple{N,Int}) where {N,T} = StdXReLU(zeros(T,n), zeros(T,n), zeros(T,n))
StdXReLU(::Type{T}, n::Int...) where {T} = StdXReLU(T, n)
StdXReLU(n::NTuple{N,Int}) where {N} = StdXReLU(Float64, n)
StdXReLU(n::Int...) = StdXReLU(n)
xReLU(l::StdXReLU) = xReLU(l.θ, Ones{eltype(l.θ)}(size(l)), l.Δ, l.ξ)
Base.size(l::StdXReLU) = size(l.θ)
Base.length(l::StdXReLU) = length(l.θ)
Base.repeat(l::StdXReLU, n::Int...) = StdXReLU(
    repeat(l.θ, n...), repeat(l.Δ, n...), repeat(l.ξ, n...)
)
effective(l::StdXReLU, inputs::AbstractArray; β::Real=1) = effective(xReLU(l), inputs; β)
energies(l::StdXReLU, x::AbstractArray) = energies(xReLU(l), x)
free_energies(l::StdXReLU) = free_energies(xReLU(l))
transfer_sample(l::StdXReLU) = transfer_sample(xReLU(l))
transfer_mode(l::StdXReLU) = transfer_mode(xReLU(l))
transfer_mean(l::StdXReLU) = transfer_mean(xReLU(l))
transfer_var(l::StdXReLU) = transfer_var(xReLU(l))
transfer_std(l::StdXReLU) = transfer_std(xReLU(l))
transfer_mean_abs(l::StdXReLU) = transfer_mean_abs(xReLU(l))

function ∂free_energy(l::StdXReLU, inputs::AbstractArray; wts = nothing)
    ∂ = ∂free_energy(xReLU(l), inputs; wts)
    return (θ = ∂.θ, Δ = ∂.Δ, ξ = ∂.ξ)
end

∂free_energy(l::StdXReLU) = ∂free_energy(l, 0)

function ∂energy(l::StdXReLU, stats::xReLUStats)
    ∂ = ∂energy(xReLU(l), stats)
    return (θ = ∂.θ, Δ = ∂.Δ, ξ = ∂.ξ)
end

suffstats(l::StdXReLU, data::AbstractArray; wts = nothing) = suffstats(xReLU(l), data; wts)
