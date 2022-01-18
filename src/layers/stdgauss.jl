"""
    StdGauss

StdGauss layer, a standardized (zero mean, unit variances) Gaussian layer.
"""
struct StdGauss{N,T} <: AbstractLayer{N}
    n::NTuple{N,Int}
    function StdGauss(::Type{T}, n::Int...) where {T}
        @assert all(n .≥ 0)
        return new{length(n), T}(n)
    end
end
StdGauss(n::Int...) = StdGauss(Float64, n...)

Base.size(layer::StdGauss) = layer.n
Base.size(layer::StdGauss, d::Int) = layer.n[d]
Base.ndims(layer::StdGauss) = length(layer.n)
Base.length(layer::StdGauss) = prod(layer.n)

function effective(layer::StdGauss{N,T}, inputs::AbstractArray; β::Real = true) where {N,T}
    @assert size(layer) == size(inputs)[1:ndims(layer)]
    θ = T(β) * inputs
    γ = oftype(θ, FillArrays.Fill(β, size(inputs)))
    return Gaussian(θ, γ)
end

function energies(layer::StdGauss, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return x.^2 / 2
end

free_energies(layer::StdGauss{N,T}) where {N,T} = FillArrays.Fill(-log(2T(π))/2, size(layer))
transfer_mode(layer::StdGauss) = FillArrays.Falses(size(layer))
transfer_mean(layer::StdGauss) = transfer_mode(layer)
transfer_var(layer::StdGauss) = FillArrays.Trues(size(layer))
transfer_std(layer::StdGauss) = transfer_var(layer)  # √1 = 1
transfer_mean_abs(layer::StdGauss{N,T}) where {N,T} = FillArrays.Fill(√(2/T(π)), size(layer))
transfer_sample(layer::StdGauss{N,T}) where {N,T} = randn(T, size(layer))
∂free_energy(::StdGauss) = (;)
∂energy(::StdGauss) = (;)

function sufficient_statistics(layer::StdGauss, x::AbstractArray; wts = nothing)
    @assert size(layer) == size(x)[1:ndims(layer)]
    @assert isnothing(wts) || size(wts) == batchdims(layer, x)
    return (;)
end
