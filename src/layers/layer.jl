abstract type AbstractLayer{N} end

Base.ndims(::AbstractLayer{N}) where {N} = N

function effective(layer::AbstractLayer, input::Real; β::Real = true)
    inputs = FillArrays.Falses(size(layer))
    return effective(layer, inputs; β)
end

"""
    energy(layer, x)

Layer energy, reduced over layer dimensions.
"""
function energy(layer::AbstractLayer{N}, x::AbstractTensor{N}) where {N}
    check_size(layer, x)
    return sum(energies(layer, x))::Number
end

function energy(layer::AbstractLayer, x::AbstractTensor{N}) where {N}
    check_size(layer, x)
    E = sum(energies(layer, x); dims = 1:ndims(layer))
    return reshape(E, size(x, N))::AbstractVector
end

function energy(layer::AbstractLayer, x::Real)::Number
    xs = FillArrays.Fill(x, size(layer))
    return energy(layer, xs)::Number
end

"""
    free_energy(layer, inputs = 0; β = 1)

Cumulant generating function of layer, reduced over layer dimensions.
"""
function free_energy(
    layer::AbstractLayer{N}, inputs::AbstractTensor{N}; β::Real = true
) where {N}
    check_size(layer, inputs)
    F = free_energies(layer, inputs; β)
    return sum(F)::Number
end

function free_energy(layer::AbstractLayer, inputs::AbstractTensor; β::Real = true)
    check_size(layer, inputs)
    F = free_energies(layer, inputs; β)
    f = sum(F; dims = 1:ndims(layer))
    return reshape(f, size(inputs, ndims(inputs)))::AbstractVector
end

function free_energy(layer::AbstractLayer, input::Real = false; β::Real = true)
    inputs = FillArrays.Fill(input, size(layer))
    return free_energy(layer, inputs; β)::Number
end

"""
    transfer_sample(layer, inputs = 0; β = 1)

Samples layer configurations conditioned on inputs.
"""
function transfer_sample(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    check_size(layer, inputs)
    layer_ = effective(layer, inputs; β)
    return transfer_sample(layer_)
end

"""
    transfer_mode(layer, inputs = 0)

Mode of unit activations.
"""
function transfer_mode(layer::AbstractLayer, inputs::Union{Real, AbstractTensor})
    check_size(layer, inputs)
    layer_ = effective(layer, inputs)
    return transfer_mode(layer_)
end

"""
    transfer_mean(layer, inputs = 0; β = 1)

Mean of unit activations.
"""
function transfer_mean(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    check_size(layer, inputs)
    layer_ = effective(layer, inputs; β)
    return transfer_mean(layer_)
end

"""
    transfer_var(layer, inputs = 0; β = 1)

Variance of unit activations.
"""
function transfer_var(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    check_size(layer, inputs)
    layer_ = effective(layer, inputs; β)
    return transfer_var(layer_)
end

"""
    transfer_std(layer, inputs = 0; β = 1)

Standard deviation of unit activations.
"""
function transfer_std(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return transfer_std(layer_)
end

"""
    transfer_mean_abs(layer, inputs = 0; β = 1)

Mean of absolute value of unit activations.
"""
function transfer_mean_abs(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return transfer_mean_abs(layer_)
end

"""
    free_energies(layer, inputs = 0; β = 1)

Cumulant generating function of units in layer (not reduced over layer dimensions).
"""
function free_energies(
    layer::AbstractLayer, inputs::Union{Real, AbstractTensor}; β::Real = true
)
    layer_ = effective(layer, inputs; β)
    return free_energies(layer_) / β
end

function energies(layer::AbstractLayer, x::Real)
    xs = FillArrays.Fill(x, size(layer))
    return energies(layer, xs)
end

"""
    flatten(layer, x)

Flattens `x` into a scalar, vector, or matrix (where last dimension is batch),
consistently with `layer` dimensions.
"""
flatten(::AbstractLayer, x::Real) = x
function flatten(layer::AbstractLayer{N}, x::AbstractTensor{N}) where {N}
    check_size(layer, x)
    @assert size(x) == size(layer)
    return reshape(x, length(layer))
end
function flatten(layer::AbstractLayer, x::AbstractTensor{N}) where {N}
    check_size(layer, x)
    @assert size(x) == (size(layer)..., size(x, N))
    return reshape(x, length(layer), size(x, N))
end

"""
    unflatten(layer, x)

Given a flattened (scalar, vector, or matrix) `x`, reshapes it into to match the
size of `layer`.
"""
unflatten(layer::AbstractLayer, x::Real) = FillArrays.Fill(x, size(layer))
function unflatten(layer::AbstractLayer, x::AbstractVector)
    @assert length(layer) == length(x)
    return reshape(x, size(layer))
end
function unflatten(layer::AbstractLayer, x::AbstractMatrix)
    @assert length(layer) == size(x, 1)
    return reshape(x, size(layer)..., size(x, 2))
end

"""
    sufficient_statistics(layer, data; [wts])

Returns a `NamedTuple` of the sufficient statistics used by the layer.
"""
function sufficient_statistics(
    layer::AbstractLayer, x::AbstractTensor; wts::Nothing = nothing
)
    check_size(layer, x)
    if ndims(layer) == ndims(x)
        return sufficient_statistics(layer, reshape(x, size(x)..., 1), wts)
    else
        return sufficient_statistics(layer, x, wts)
    end
end

function check_size(layer::AbstractLayer{N}, x::AbstractTensor{N}) where {N}
    size(x) == size(layer) || throw_size_mismatch_error(layer, x)
end

function check_size(layer::AbstractLayer, x::AbstractTensor{N}) where {N}
    size(x) == (size(layer)..., size(x, N)) || throw_size_mismatch_error(layer, x)
end

check_size(::AbstractLayer, ::Real) = true

function throw_size_mismatch_error(layer::AbstractLayer, x::AbstractTensor)
    throw(DimensionMismatch(
        "size(layer)=$(size(layer)) inconsistent with size(x)=$(size(x))"
    ))
    return false
end
