# Like Potts, but uses Gumbel trick which is GPU-friendly
struct PottsGumbel{N,A} <: AbstractLayer{N}
    par::A
    function PottsGumbel{N,A}(par::A) where {N,A<:AbstractArray}
        @assert size(par, 1) == 1 # θ
        @assert ndims(par) == N + 1
        return new(par)
    end
end

PottsGumbel(par::AbstractArray) = PottsGumbel{ndims(par) - 1, typeof(par)}(par)

function PottsGumbel(; θ)
    par = vstack((θ,))
    return PottsGumbel(par)
end

PottsGumbel(::Type{T}, sz::Dims) where {T} = PottsGumbel(; θ = zeros(T, sz))
PottsGumbel(sz::Dims) = PottsGumbel(Float64, sz)

PottsGumbel(layer::Potts) = PottsGumbel(layer.par)
Potts(layer::PottsGumbel) = Potts(layer.par)

cgfs(layer::PottsGumbel, inputs = 0) = cgfs(Potts(layer), inputs)
mean_from_inputs(layer::PottsGumbel, inputs = 0) = mean_from_inputs(Potts(layer), inputs)
mean_abs_from_inputs(layer::PottsGumbel, inputs = 0) = mean_from_inputs(layer, inputs)
std_from_inputs(layer::PottsGumbel, inputs = 0) = std_from_inputs(Potts(layer), inputs)
mode_from_inputs(layer::PottsGumbel, inputs = 0) = mode_from_inputs(Potts(layer), inputs)
var_from_inputs(layer::PottsGumbel, inputs = 0) = var_from_inputs(Potts(layer), inputs)
meanvar_from_inputs(layer::PottsGumbel, inputs = 0) = meanvar_from_inputs(Potts(layer), inputs)

# This is the only change with respect to Potts. Here, we use the Gumbel trick.
function sample_from_inputs(layer::PottsGumbel, inputs = 0)
    c = categorical_sample_from_logits_gumbel(layer.θ .+ inputs)
    return onehot_encode(c, 1:size(layer, 1))
end

energies(layer::PottsGumbel, x::AbstractArray) = energies(Potts(layer), x)
energy(layer::PottsGumbel, x::AbstractArray) = energy(Potts(layer), x)
∂cgfs(layer::PottsGumbel, inputs = 0) = ∂cgfs(Potts(layer), inputs)
∂energy_from_moments(layer::PottsGumbel, moments::AbstractArray) = ∂energy_from_moments(Potts(layer), moments)
moments_from_samples(layer::PottsGumbel, data::AbstractArray; wts = nothing) = moments_from_samples(Potts(layer), data; wts)
colors(layer::PottsGumbel) = size(layer, 1)
sitedims(layer::PottsGumbel) = ndims(layer) - 1
sitesize(layer::PottsGumbel) = size(layer)[2:end]

function anneal(init::PottsGumbel, final::PottsGumbel; β::Real)
    θ = (1 - β) * init.θ + β * final.θ
    return PottsGumbel(; θ)
end

anneal_zero(l::PottsGumbel) = PottsGumbel(; θ = zero(l.θ))

function initialize!(layer::PottsGumbel, data::AbstractArray; ϵ::Real=1e-6, wts=nothing)
    PottsGumbel(initialize!(Potts(layer), data; ϵ, wts))
end

function potts_to_gumbel(layer::AbstractLayer)
    if layer isa Potts
        return PottsGumbel(layer)
    else
        return layer
    end
end

function gumbel_to_potts(layer::AbstractLayer)
    if layer isa PottsGumbel
        return Potts(layer)
    else
        return layer
    end
end
