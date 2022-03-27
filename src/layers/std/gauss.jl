struct StdGauss{N,T} <: AbstractLayer{N}
    size::NTuple{N,Int}
    function StdGauss(::Type{T}, size::NTuple{N,Int}) where {T,N}
        @assert all(≥(0), size)
        return new{N,T}(size)
    end
end

"""
    StdGauss([T], n...)

Standardized Gaussian layer, with zero mean and unit variance.
"""
StdGauss(::Type{T}, n::Int...) where {T} = StdGauss(T, n)
StdGauss(n::Int...) = StdGauss(n)
StdGauss(n::NTuple{N,Int}) where {N} = StdGauss(Float64, n)
Gaussian(l::StdGauss{N,T}) where {N,T} = Gaussian(Zeros{T}(size(l)), Ones{T}(size(l)))

Base.size(layer::StdGauss) = layer.size
Base.length(layer::StdGauss) = prod(size(layer))
Base.repeat(l::StdGauss{N,T}, n::Int...) where {N,T} = StdGauss(T, repeat_size(size(l), n...))

effective(l::StdGauss, inputs::AbstractArray; β::Real = 1) = effective(Gaussian(l), inputs; β)
energies(l::StdGauss, x::AbstractArray) = energies(Gaussian(l), x)
free_energies(layer::StdGauss) = free_energies(Gaussian(layer))
transfer_sample(layer::StdGauss) = transfer_sample(Gaussian(layer))
transfer_mode(layer::StdGauss{N,T}) where {N,T} = Zeros{T}(size(layer))
transfer_mean(layer::StdGauss) = transfer_mode(layer)
transfer_var(layer::StdGauss{N,T}) where {N,T} = Ones{T}(size(layer))
transfer_std(layer::StdGauss) = transfer_var(layer)
transfer_mean_abs(layer::StdGauss) = transfer_mean_abs(Gaussian(layer))
∂free_energy(::StdGauss) = (;)
∂energy(::StdGauss) = (;)

function ∂free_energy(layer::StdGauss, inputs::AbstractArray; wts = nothing)
    @assert size(layer) == size(inputs)[1:ndims(layer)]
    return ∂free_energy(layer)
end

struct StdGaussStats end
suffstats(::StdGauss, ::AbstractArray; wts = nothing) = StdGaussStats()
∂energy(layer::StdGauss, ::StdGaussStats) = ∂energy(layer)
