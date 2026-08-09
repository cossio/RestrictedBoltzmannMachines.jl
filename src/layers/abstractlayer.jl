abstract type AbstractLayer{N} end

_validate_layer_parameters(::AbstractLayer) = nothing

Base.ndims(::AbstractLayer{N}) where {N} = N
Base.size(layer::AbstractLayer) = Base.tail(size(getfield(layer, :par)))
# As for arrays, trailing dimensions are 1, while d < 1 is an error.
function Base.size(layer::AbstractLayer, d::Int)
    d ≥ 1 || throw(ArgumentError("dimension must be ≥ 1, got $d"))
    return size(getfield(layer, :par), d + 1)
end
Base.length(layer::AbstractLayer) = length(getfield(layer, :par)) ÷ size(getfield(layer, :par), 1)

"""
    @declare_layer Layer (θ = zeros, γ = ones)

Declares a layer type `Layer` whose named parameters are the rows of a shared `par` array,
in the given order, with the given default initializers. Generates the struct (with `par`
size validation and a `_validate_layer_parameters` hook in the inner constructor), the
`Layer(par)`, keyword, `Layer(T, sz)`, and `Layer(sz)` constructors, `Base.propertynames`,
the `Base.getproperty` accessors returning views into `par`, and the `_construct_like`
trait used by generic functions such as `anneal`.
"""
macro declare_layer(Layer, params)
    Layer isa Symbol || error("expected a layer name, got $Layer")
    Meta.isexpr(params, :tuple) && !isempty(params.args) || error("expected a (name = init, ...) tuple of parameters, got $params")
    names = Symbol[]
    inits = Any[]
    for p in params.args
        Meta.isexpr(p, :(=), 2) && p.args[1] isa Symbol || error("expected name = init, got $p")
        push!(names, p.args[1])
        push!(inits, p.args[2])
    end
    nparams = length(names)

    getproperty_body = :(return getfield(layer, name))
    for i in nparams:-1:1
        value = if nparams == 1
            # dropdims instead of a view; see https://github.com/JuliaGPU/CUDA.jl/issues/1957
            :(return dropdims(getfield(layer, :par); dims = 1))
        else
            :(return @view getfield(layer, :par)[$i, ..])
        end
        getproperty_body = Expr(i == 1 ? :if : :elseif, :(name === $(QuoteNode(names[i]))), value, getproperty_body)
    end

    defaults = [Expr(:kw, name, :($init(T, sz))) for (name, init) in zip(names, inits)]

    # Stack the parameters as rows of `par` via vcat of reshapes rather than
    # `stack`: unlike `stack`, vcat stays inferable when the arguments mix
    # container types (e.g. a view of `par` alongside a broadcast result) and
    # has ChainRules rules, so Zygote can differentiate through layer
    # conversions such as dReLU(::pReLU) that call this constructor.
    rows = [:(reshape($name, 1, size($name)...)) for name in names]

    return esc(
        quote
            Base.@__doc__ struct $Layer{N, A} <: AbstractLayer{N}
                par::A
                function $Layer{N, A}(par::A) where {N, A <: AbstractArray}
                    @assert size(par, 1) == $nparams
                    @assert ndims(par) == N + 1
                    layer = new(par)
                    _validate_layer_parameters(layer)
                    return layer
                end
            end

            $Layer(par::AbstractArray) = $Layer{ndims(par) - 1, typeof(par)}(par)
            $Layer(; $(names...)) = $Layer(vcat($(rows...)))
            $Layer(::Type{T}, sz::Dims) where {T} = $Layer(; $(defaults...))
            $Layer(sz::Dims) = $Layer(Float64, sz)

            _construct_like(::$Layer, par::AbstractArray) = $Layer(par)

            Base.propertynames(::$Layer) = ($(QuoteNode.(names)...),)

            function Base.getproperty(layer::$Layer, name::Symbol)
                $getproperty_body
            end
        end
    )
end

"""
    flatten(layer, x)

Returns a vectorized version of `x`.
"""
function flatten(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    if ndims(layer) == ndims(x)
        return vec(x)
    else
        return reshape(x, length(layer), prod(size(x, d) for d in (ndims(layer) + 1):ndims(x)))
    end
end

"""
    energy(layer, x)

Layer energy, reduced over layer dimensions.
"""
function energy(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    Es = energies(layer, x)
    if ndims(layer) == ndims(x)
        return sum(Es)
    else
        E = sum(Es; dims = 1:ndims(layer))
        return reshape(E, size(x)[(ndims(layer) + 1):end])
    end
end

"""
    cgf(layer, [inputs])

Cumulant generating function of layer, reduced over layer dimensions.
"""
function cgf(layer::AbstractLayer, inputs::AbstractArray = Falses(size(layer)))
    Γ = cgfs(layer, inputs)
    if ndims(layer) == ndims(inputs)
        return sum(Γ)
    else
        _Γ = sum(Γ; dims = 1:ndims(layer))
        return reshape(_Γ, size(inputs)[(ndims(layer) + 1):end])
    end
end

"""
    std_from_inputs(layer, [inputs])

Standard deviation of unit activations from inputs.
"""
std_from_inputs(layer::AbstractLayer, inputs::AbstractArray = Falses(size(layer))) = sqrt.(var_from_inputs(layer, inputs))

"""
    meanvar_from_inputs(layer, [inputs])

Mean and variance of unit activations from inputs.
"""
function meanvar_from_inputs(layer::AbstractLayer, inputs::AbstractArray = Falses(size(layer)))
    return (mean_from_inputs(layer, inputs), var_from_inputs(layer, inputs))
end

"""
    batchdims(layer, x)

Indices of batch dimensions in `x`, with respect to `layer`.
"""
function batchdims(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return (ndims(layer) + 1):ndims(x)
end

"""
    batch_size(layer, x)

Batch sizes of `x`, with respect to `layer`. A scalar `x` broadcasts over the
layer and has no batch dimensions.
"""
function batch_size(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return size(x)[batchdims(layer, x)]
end
batch_size(::AbstractLayer, ::Number) = ()

"""
    uniform_weights(layer, x)

Lazy uniform weights over the batch dimensions of `x`.
"""
uniform_weights(layer::AbstractLayer, x) = Trues(batch_size(layer, x))

"""
    batchmean(layer, x; wts = uniform_weights(layer, x))

Mean of `x` over batch dimensions, weigthed by `wts`.
"""
function batchmean(
        layer::AbstractLayer, x::AbstractArray; wts::AbstractArray = uniform_weights(layer, x)
    )
    @assert size(wts) == batch_size(layer, x)
    return wmean(x; wts)
end

"""
    batchvar(layer, x; wts = uniform_weights(layer, x), [mean])

Variance of `x` over batch dimensions, weigthed by `wts`.
"""
function batchvar(
        layer::AbstractLayer, x::AbstractArray;
        wts::AbstractArray = uniform_weights(layer, x),
        mean::AbstractArray = batchmean(layer, x; wts)
    )
    return batchmean(layer, (x .- mean) .^ 2; wts)
end

"""
    batchstd(layer, x; wts = uniform_weights(layer, x), [mean])

Standard deviation of `x` over batch dimensions, weigthed by `wts`.
"""
function batchstd(
        layer::AbstractLayer, x::AbstractArray;
        wts::AbstractArray = uniform_weights(layer, x),
        mean::AbstractArray = batchmean(layer, x; wts)
    )
    return sqrt.(batchvar(layer, x; wts, mean))
end

"""
    batchcov(layer, x; wts = uniform_weights(layer, x), [mean])

Covariance of `x` over batch dimensions, weigthed by `wts`.
"""
function batchcov(
        layer::AbstractLayer, x::AbstractArray;
        wts::AbstractArray = uniform_weights(layer, x),
        mean::AbstractArray = batchmean(layer, x; wts)
    )
    @assert size(wts) == batch_size(layer, x)
    # reshape into a matrix even for unbatched `x`, where `flatten` gives a vector
    ξ = reshape(flatten(layer, x .- mean), length(layer), :)
    C = _weighted_outer(ξ, wts, ξ) / sum(wts)
    return reshape(C, size(layer)..., size(layer)...)
end

"""
    total_mean_from_inputs(layer, [inputs]; wts = uniform_weights(layer, inputs))

Total mean of unit activations from inputs.
"""
function total_mean_from_inputs(
        layer::AbstractLayer, inputs::AbstractArray = Falses(size(layer));
        wts::AbstractArray = uniform_weights(layer, inputs)
    )
    h_ave = mean_from_inputs(layer, inputs)
    return batchmean(layer, h_ave; wts)
end

"""
    total_var_from_inputs(layer, [inputs]; wts = uniform_weights(layer, inputs))

Total variance of unit activations from inputs.
"""
function total_var_from_inputs(
        layer::AbstractLayer, inputs::AbstractArray = Falses(size(layer));
        wts::AbstractArray = uniform_weights(layer, inputs)
    )
    h_ave, h_var = meanvar_from_inputs(layer, inputs)
    ν_int = batchmean(layer, h_var; wts) # intrinsic noise
    ν_ext = batchvar(layer, h_ave; wts) # extrinsic noise
    return ν_int + ν_ext # law of total variance
end

"""
    total_meanvar_from_inputs(layer, [inputs]; wts = uniform_weights(layer, inputs))

Total mean and total variance of unit activations from inputs.
"""
function total_meanvar_from_inputs(
        layer::AbstractLayer, inputs::AbstractArray = Falses(size(layer));
        wts::AbstractArray = uniform_weights(layer, inputs)
    )
    h_ave, h_var = meanvar_from_inputs(layer, inputs)
    μ = batchmean(layer, h_ave; wts)
    ν_int = batchmean(layer, h_var; wts) # intrinsic noise
    ν_ext = batchvar(layer, h_ave; wts, mean = μ) # extrinsic noise
    ν = ν_int + ν_ext # law of total variance
    return (μ = μ, ν = ν)
end

"""
    moments_from_samples(layer, data; wts = uniform_weights(layer, data))

Empirical moments of `data`, batch-averaged with weights `wts`. Each layer
defines which moments it computes (see the docstrings of its specific
methods); generally they are the sufficient statistics of the layer
distribution, which do not depend on the layer parameters, so they can be
computed once from a dataset and reused as the parameters change (see `pcd!`).
The first axis indexes the moment and the remaining axes are `size(layer)`
(batch dimensions of `data` are averaged over). The number of moments need
not match the number of parameters (e.g. `nsReLU` uses the 4-slot dReLU
layout while having 3 parameters).

`moments_from_inputs` returns conditional moments in this same layout, and
`∂energy_from_moments` consumes it.
"""
function moments_from_samples end

"""
    ∂energy_from_moments(layer, moments)

Derivative of the layer's mean energy with respect to its parameters, evaluated
at a moments array (see `moments_from_samples` for the layout). The first axis
of the result indexes the parameter, as in `layer.par`, and trailing batch
dimensions of `moments` are preserved. Since the energy is linear in the
sufficient statistics, this is a linear map of `moments`, with coefficients
that may depend on the current parameters.
"""
function ∂energy_from_moments end

"""
    ∂energy(layer, data; wts = uniform_weights(layer, data))

Derivative of average energy of `data` with respect to `layer` parameters.
"""
function ∂energy(
        layer::AbstractLayer, data::AbstractArray;
        wts::AbstractArray = uniform_weights(layer, data)
    )
    moments = moments_from_samples(layer, data; wts)
    return ∂energy_from_moments(layer, moments)
end

"""
    moments_from_inputs(layer, [inputs])

Moments of the unit activations under the conditional distribution given `inputs`,
in the same layout as `moments_from_samples`: the first axis indexes the
moment, the next axes are `size(layer)`, and any trailing batch dimensions of
`inputs` are preserved (the moments are per-configuration, not batch-averaged).
"""
function moments_from_inputs end

"""
    batchmean_moments(layer, moments; wts = uniform_weights(layer, view(moments, 1, ..)))

Average a per-configuration moments array (as returned by `moments_from_inputs`
with batched inputs) over its batch dimensions, weighted by `wts` (lazy uniform
weights by default).
"""
function batchmean_moments(
        layer::AbstractLayer, moments::AbstractArray;
        wts::AbstractArray = uniform_weights(layer, view(moments, 1, ..))
    )
    @assert ndims(wts) == ndims(moments) - ndims(layer) - 1
    return wmean(moments; wts)
end

"""
    mean_from_moments(layer, moments)

Mean unit activations `<x>` from a moments array (see `moments_from_samples`
for the layout). Batch dimensions of `moments` are preserved.
"""
function mean_from_moments end

"""
    var_from_moments(layer, moments)

Variance of unit activations from a moments array (see `moments_from_samples`
for the layout). Batch dimensions of `moments` are preserved.
"""
function var_from_moments end

"""
    ∂cgfs(layer, [inputs])

Gradient of `cgfs` with respect to the layer parameters, for each configuration of
`inputs` (batch dimensions are preserved; the first axis indexes the parameter, as
in `layer.par`). Since the cumulant generating function and the energy are conjugate,
this is `-∂energy_from_moments` evaluated at the conditional moments given `inputs`.
"""
function ∂cgfs(layer::AbstractLayer, inputs::AbstractArray = Falses(size(layer)))
    return -∂energy_from_moments(layer, moments_from_inputs(layer, inputs))
end

"""
    ∂cgf(layer, [inputs]; wts = uniform_weights(layer, inputs))

Unit activation moments, conjugate to layer parameters.
These are obtained by differentiating `cgfs` with respect to the layer parameters.
Averages over configurations (weigthed by `wts`).
"""
function ∂cgf(
        layer::AbstractLayer, inputs::AbstractArray = Falses(size(layer));
        wts::AbstractArray = uniform_weights(layer, inputs)
    )
    ∂Fs = ∂cgfs(layer, inputs)
    @assert ndims(wts) == ndims(∂Fs) - ndims(layer.par)
    return wmean(∂Fs; wts)
end
