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

# From common.jl

Base.size(layer::PottsGumbel) = size(layer.θ)
Base.length(layer::PottsGumbel) = length(layer.θ)
Base.propertynames(::PottsGumbel) = (:θ,)

function Base.getproperty(layer::PottsGumbel, name::Symbol)
    if name === :θ
        # https://github.com/JuliaGPU/CUDA.jl/issues/1957
        # return @view getfield(layer, :par)[1, ..]
        return dropdims(getfield(layer, :par); dims=1)
    else
        return getfield(layer, name)
    end
end

energies(layer::PottsGumbel, x::AbstractArray) = energies(Potts(layer), x)
energy(layer::PottsGumbel, x::AbstractArray) = energy(Potts(layer), x)
∂cgfs(layer::PottsGumbel, inputs = 0) = ∂cgfs(Potts(layer), inputs)
∂energy_from_moments(layer::PottsGumbel, moments::AbstractArray) = ∂energy_from_moments(Potts(layer), moments)
moments_from_samples(layer::PottsGumbel, data::AbstractArray; wts = nothing) = moments_from_samples(Potts(layer), data; wts)
colors(layer::PottsGumbel) = size(layer, 1)
sitedims(layer::PottsGumbel) = ndims(layer) - 1
sitesize(layer::PottsGumbel) = size(layer)[2:end]

# other stuff

grad2ave(l::PottsGumbel, ∂::AbstractArray) = grad2ave(Potts(l), ∂)

function anneal(init::PottsGumbel, final::PottsGumbel; β::Real)
    θ = (1 - β) * init.θ + β * final.θ
    return PottsGumbel(; θ)
end

anneal_zero(l::PottsGumbel) = PottsGumbel(; θ = zero(l.θ))

function substitution_matrix_sites(rbm::RBM{<:PottsGumbel}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex})
    substitution_matrix_sites(RBM(Potts(rbm.visible), rbm.hidden, rbm.w), v, sites)
end

function substitution_matrix_exhaustive(rbm::RBM{<:PottsGumbel}, v::AbstractArray)
    substitution_matrix_exhaustive(RBM(Potts(rbm.visible), rbm.hidden, rbm.w), v)
end

function ∂regularize_fields!(∂::AbstractArray, layer::PottsGumbel; l2_fields::Real = 0)
    ∂regularize_fields!(∂, Potts(layer); l2_fields )
end

rescale_activations!(layer::PottsGumbel, λ::AbstractArray) = false

function initialize!(layer::PottsGumbel, data::AbstractArray; ϵ::Real=1e-6, wts=nothing)
    PottsGumbel(initialize!(Potts(layer), data; ϵ, wts))
end

function initialize!(layer::PottsGumbel)
    layer.θ .= 0
    return layer
end

"""
    potts_to_gumbel(rbm)

Converts Potts layers to PottsGumbel layers.
"""
function potts_to_gumbel(rbm::RBM)
    visible = potts_to_gumbel(rbm.visible)
    hidden = potts_to_gumbel(rbm.hidden)
    return RBM(visible, hidden, rbm.w)
end

function potts_to_gumbel(rbm::StandardizedRBM)
    visible = potts_to_gumbel(rbm.visible)
    hidden = potts_to_gumbel(rbm.hidden)
    return StandardizedRBM(visible, hidden, rbm.w, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
end

"""
    gumbel_to_potts(rbm)

Converts PottsGumbel layers to Potts layers.
"""
function gumbel_to_potts(rbm::RBM)
    visible = gumbel_to_potts(rbm.visible)
    hidden = gumbel_to_potts(rbm.hidden)
    return RBM(visible, hidden, rbm.w)
end

function gumbel_to_potts(rbm::StandardizedRBM)
    visible = gumbel_to_potts(rbm.visible)
    hidden = gumbel_to_potts(rbm.hidden)
    return StandardizedRBM(visible, hidden, rbm.w, rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h)
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

function zerosum(rbm::RBM{<:PottsGumbel, <:PottsGumbel})
    return potts_to_gumbel(zerosum(gumbel_to_potts(rbm)))
end

function zerosum(rbm::RBM{<:PottsGumbel,<:AbstractLayer})
    _rbm = zerosum(gumbel_to_potts(rbm))
    return RBM(PottsGumbel(_rbm.visible), _rbm.hidden, _rbm.w)
end

function zerosum(rbm::RBM{<:AbstractLayer,<:PottsGumbel})
    _rbm = zerosum(gumbel_to_potts(rbm))
    return RBM(_rbm.visible, PottsGumbel(_rbm.hidden), _rbm.w)
end

function zerosum!(rbm::RBM{<:PottsGumbel,<:PottsGumbel})
    return potts_to_gumbel(zerosum!(gumbel_to_potts(rbm)))
end

function zerosum!(rbm::RBM{<:PottsGumbel,<:AbstractLayer})
    _rbm = zerosum!(gumbel_to_potts(rbm))
    return RBM(PottsGumbel(_rbm.visible), _rbm.hidden, _rbm.w)
end

function zerosum!(rbm::RBM{<:AbstractLayer,<:PottsGumbel})
    _rbm = zerosum!(gumbel_to_potts(rbm))
    return RBM(_rbm.visible, PottsGumbel(_rbm.hidden), _rbm.w)
end

zerosum!(∂::∂RBM, rbm::RBM{<:PottsGumbel,<:PottsGumbel}) = zerosum!(∂, gumbel_to_potts(rbm))
zerosum!(∂::∂RBM, rbm::RBM{<:AbstractLayer,<:PottsGumbel}) = zerosum!(∂, gumbel_to_potts(rbm))
zerosum!(∂::∂RBM, rbm::RBM{<:PottsGumbel,<:AbstractLayer}) = zerosum!(∂, gumbel_to_potts(rbm))

zerosum_weights(weights::AbstractArray, rbm::RBM{<:PottsGumbel,<:PottsGumbel}) = zerosum_weights(weights, gumbel_to_potts(rbm))
zerosum_weights(weights::AbstractArray, rbm::RBM{<:AbstractLayer,<:PottsGumbel}) = zerosum_weights(weights, gumbel_to_potts(rbm))
zerosum_weights(weights::AbstractArray, rbm::RBM{<:PottsGumbel,<:AbstractLayer}) = zerosum_weights(weights, gumbel_to_potts(rbm))
