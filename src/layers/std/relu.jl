struct StdReLU{N,T,A<:AbstractArray{T,N}} <: AbstractLayer{N}
    θ::A
end

"""
    StdReLU([T], n...)

Standardized ReLU layer, with unit scale parameters.
"""
StdReLU(::Type{T}, n::NTuple{N,Int}) where {N,T} = StdReLU(zeros(T, n))
StdReLU(::Type{T}, n::Int...) where {T} = StdReLU(T, n)
StdReLU(n::NTuple{N,Int}) where {N} = StdReLU(Float64, n)
StdReLU(n::Int...) = StdReLU(n)
ReLU(l::StdReLU) = ReLU(l.θ, Ones{eltype(l.θ)}(size(l)))

Base.size(layer::StdReLU) = size(layer.θ)
Base.length(layer::StdReLU) = length(layer.θ)
Base.repeat(l::StdReLU, n::Int...) = StdReLU(repeat(l.θ, n...))
effective(l::StdReLU, inputs::AbstractArray; β::Real = 1) = effective(ReLU(l), inputs; β)
energies(layer::StdReLU, x::AbstractArray) = energies(ReLU(layer), x)
free_energies(layer::StdReLU) = free_energies(ReLU(layer))
transfer_sample(layer::StdReLU) = transfer_sample(ReLU(layer))
transfer_mode(layer::StdReLU) = max.(layer.θ, 0)
transfer_mean(layer::StdReLU) = layer.θ .+ tnmean.(-layer.θ)
transfer_var(layer::StdReLU) = tnvar.(-layer.θ)
transfer_std(layer::StdReLU) = sqrt.(transfer_var(layer))
transfer_mean_abs(layer::StdReLU) = transfer_mean(layer)
∂free_energy(layer::StdReLU) = (; θ = -transfer_mean(layer))

function ∂free_energy(layer::StdReLU, inputs::AbstractArray; wts = nothing)
    @assert size(layer) == size(inputs)[1:ndims(layer)]
    layer_eff = StdReLU(layer.θ .+ inputs)
    ∂Feff = ∂free_energy(layer_eff)
    return (; θ = batchmean(layer, ∂Feff.θ; wts))
end

struct StdReLUStats{A<:AbstractArray}
    x::A
    function StdReLUStats(layer::AbstractLayer, data::AbstractArray; wts = nothing)
        @assert size(layer) == size(data)[1:ndims(layer)]
        x = batchmean(layer, data; wts)
        return new{typeof(x)}(x)
    end
end

Base.size(stats::StdReLUStats) = size(stats.x)
suffstats(l::StdReLU, data::AbstractArray; wts = nothing) = StdReLUStats(l, data; wts)

function ∂energy(layer::StdReLU, stats::StdReLUStats)
    @assert size(layer) == size(stats)
    return (; θ = -stats.x)
end
