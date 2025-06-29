#= nsReLU : a variant of dReLU units without scale parameter γ (which is fixed at 1) =#

"""
    nsReLU

A variant of `dReLU` units without scale parameter γ (which is fixed at 1). This is done
to remove the gauge invariance between the weights and the hidden units scale.
"""
struct nsReLU{N,A} <: RestrictedBoltzmannMachines.AbstractLayer{N}
    par::A
    function nsReLU{N,A}(par::A) where {N,A<:AbstractArray}
        @assert size(par, 1) == 3 # θ, Δ, ξ (there is no γ)
        @assert ndims(par) == N + 1
        return new(par)
    end
end

nsReLU(par::AbstractArray) = nsReLU{ndims(par) - 1, typeof(par)}(par)
nsReLU(; θ, Δ, ξ) = nsReLU(vstack((θ, Δ, ξ)))
nsReLU(::Type{T}, sz::Dims) where {T} = nsReLU(; θ=zeros(T, sz), Δ=zeros(T, sz), ξ=zeros(T, sz))
nsReLU(sz::Dims) = nsReLU(Float64, sz)

Base.propertynames(::nsReLU) = (:θ, :Δ, :ξ)

function Base.getproperty(layer::nsReLU, name::Symbol)
    if name === :θ
        return @view getfield(layer, :par)[1, ..]
    elseif name === :Δ
        return @view getfield(layer, :par)[2, ..]
    elseif name === :ξ
        return @view getfield(layer, :par)[3, ..]
    else
        return getfield(layer, name)
    end
end

energies(layer::nsReLU, x::AbstractArray) = energies(xReLU(layer), x)
cgfs(layer::nsReLU, inputs = 0) = cgfs(xReLU(layer), inputs)
sample_from_inputs(layer::nsReLU, inputs = 0) = sample_from_inputs(xReLU(layer), inputs)
mode_from_inputs(layer::nsReLU, inputs = 0) = mode_from_inputs(xReLU(layer), inputs)
mean_from_inputs(layer::nsReLU, inputs = 0) = mean_from_inputs(xReLU(layer), inputs)
var_from_inputs(layer::nsReLU, inputs = 0) = var_from_inputs(xReLU(layer), inputs)
meanvar_from_inputs(layer::nsReLU, inputs = 0) = meanvar_from_inputs(xReLU(layer), inputs)
std_from_inputs(layer::nsReLU, inputs = 0) = sqrt.(var_from_inputs(layer, inputs))
mean_abs_from_inputs(layer::nsReLU, inputs = 0) = mean_abs_from_inputs(xReLU(layer), inputs)

function ∂cgfs(layer::nsReLU, inputs = 0)
    xrelu = xReLU(layer)
    ∂ = ∂cgfs(xrelu, inputs)
    return ∂[[1,3,4], ..] # skip γ
end

function ∂energy_from_moments(layer::nsReLU, moments::AbstractArray)
    ∂ = ∂energy_from_moments(xReLU(layer), moments)
    return ∂[[1,3,4], ..] # skip γ
    @assert size(layer.par) == size(moments)
end

xReLU(layer::nsReLU) = xReLU(; layer.θ, γ=ones(size(layer.θ)), layer.Δ, layer.ξ)
dReLU(layer::nsReLU) = dReLU(xReLU(layer))

function initialize!(layer::nsReLU, data::AbstractArray; wts=nothing)
    @assert size(layer) == size(data)[1:ndims(layer)]
    μ = batchmean(layer, data; wts)
    layer.θ .= μ
    layer.Δ .= layer.ξ .= 0
    return layer
end

function initialize!(layer::nsReLU)
    layer.θ .= layer.Δ .= layer.ξ .= 0
    return layer
end

function ∂regularize_fields(layer::nsReLU; l2_fields::Real = 0)
    ∂θ = l2_fields * layer.θ
    ∂Δ = zero(layer.Δ)
    ∂ξ = zero(layer.ξ)
    return vstack((∂θ, ∂Δ, ∂ξ))
end

shift_fields(l::nsReLU, a::AbstractArray) = nsReLU(; θ = l.θ .+ a, l.Δ, l.ξ)

gpu(layer::nsReLU) = nsReLU(gpu(layer.par))
cpu(layer::nsReLU) = nsReLU(cpu(layer.par))
