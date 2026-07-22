abstract type AbstractLayer{N} end

_validate_layer_parameters(::AbstractLayer) = nothing

Base.ndims(::AbstractLayer{N}) where {N} = N
Base.size(layer::AbstractLayer) = Base.tail(size(getfield(layer, :par)))
Base.size(layer::AbstractLayer, d::Int) = size(layer)[d]
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
            $Layer(; $(names...)) = $Layer(vstack(($(names...),)))
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
    cgf(layer, inputs = 0)

Cumulant generating function of layer, reduced over layer dimensions.
"""
function cgf(layer::AbstractLayer, inputs = 0)
    Γ = cgfs(layer, inputs)
    if inputs isa Real || ndims(layer) == ndims(inputs)
        return sum(Γ)
    else
        _Γ = sum(Γ; dims = 1:ndims(layer))
        return reshape(_Γ, size(inputs)[(ndims(layer) + 1):end])
    end
end

"""
    std_from_inputs(layer, inputs = 0)

Standard deviation of unit activations from inputs.
"""
std_from_inputs(layer::AbstractLayer, inputs = 0) = sqrt.(var_from_inputs(layer, inputs))

"""
    meanvar_from_inputs(layer, inputs = 0)

Mean and variance of unit activations from inputs.
"""
function meanvar_from_inputs(layer::AbstractLayer, inputs = 0)
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

Batch sizes of `x`, with respect to `layer`.
"""
function batch_size(layer::AbstractLayer, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return size(x)[batchdims(layer, x)]
end

"""
    batchmean(layer, x; wts = nothing)

Mean of `x` over batch dimensions, weigthed by `wts`.
"""
function batchmean(layer::AbstractLayer, x::AbstractArray; wts = nothing)
    @assert size(layer) == size(x)[1:ndims(layer)]
    if ndims(layer) == ndims(x)
        wts::Nothing
        return x
    else
        μ = wmean(x; wts, dims = batchdims(layer, x))
        return reshape(μ, size(layer))
    end
end

"""
    batchvar(layer, x; wts = nothing, [mean])

Variance of `x` over batch dimensions, weigthed by `wts`.
"""
function batchvar(
        layer::AbstractLayer, x::AbstractArray; wts = nothing, mean = batchmean(layer, x; wts)
    )
    @assert size(layer) == size(x)[1:ndims(layer)] == size(mean)
    return batchmean(layer, (x .- mean) .^ 2; wts)
end

"""
    batchstd(layer, x; wts = nothing, [mean])

Standard deviation of `x` over batch dimensions, weigthed by `wts`.
"""
function batchstd(
        layer::AbstractLayer, x::AbstractArray; wts = nothing, mean = batchmean(layer, x; wts)
    )
    return sqrt.(batchvar(layer, x; wts, mean))
end

"""
    batchcov(layer, x; wts = nothing, [mean])

Covariance of `x` over batch dimensions, weigthed by `wts`.
"""
function batchcov(
        layer::AbstractLayer, x::AbstractArray; wts = nothing, mean = batchmean(layer, x; wts)
    )
    @assert size(layer) == size(x)[1:ndims(layer)] == size(mean)
    ξ = flatten(layer, x .- mean)
    if isnothing(wts)
        C = ξ * ξ' / size(ξ, 2)
    else
        @assert size(wts) == batch_size(layer, x)
        w = Diagonal(vec(wts))
        C = ξ * w * ξ' / sum(w)
    end
    return reshape(C, size(layer)..., size(layer)...)
end

"""
    total_mean_from_inputs(layer, inputs; wts = nothing)

Total mean of unit activations from inputs.
"""
function total_mean_from_inputs(layer::AbstractLayer, inputs = 0; wts = nothing)
    h_ave = mean_from_inputs(layer, inputs)
    return batchmean(layer, h_ave; wts)
end

"""
    total_var_from_inputs(layer, inputs; wts = nothing)

Total variance of unit activations from inputs.
"""
function total_var_from_inputs(layer::AbstractLayer, inputs = 0; wts = nothing)
    h_ave, h_var = meanvar_from_inputs(layer, inputs)
    ν_int = batchmean(layer, h_var; wts) # intrinsic noise
    ν_ext = batchvar(layer, h_ave; wts) # extrinsic noise
    return ν_int + ν_ext # law of total variance
end

"""
    total_meanvar_from_inputs(layer, inputs; wts = nothing)

Total mean and total variance of unit activations from inputs.
"""
function total_meanvar_from_inputs(layer::AbstractLayer, inputs = 0; wts = nothing)
    h_ave, h_var = meanvar_from_inputs(layer, inputs)
    μ = batchmean(layer, h_ave; wts)
    ν_int = batchmean(layer, h_var; wts) # intrinsic noise
    ν_ext = batchvar(layer, h_ave; wts, mean = μ) # extrinsic noise
    ν = ν_int + ν_ext # law of total variance
    return (μ = μ, ν = ν)
end

"""
    ∂energy(layer, data; wts = nothing)

Derivative of average energy of `data` with respect to `layer` parameters.
"""
function ∂energy(layer::AbstractLayer, data::AbstractArray; wts = nothing)
    moments = moments_from_samples(layer, data; wts)
    return ∂energy_from_moments(layer, moments)
end

"""
    ∂cgf(layer, inputs = 0; wts = 1)

Unit activation moments, conjugate to layer parameters.
These are obtained by differentiating `cgfs` with respect to the layer parameters.
Averages over configurations (weigthed by `wts`).
"""
function ∂cgf(layer::AbstractLayer, inputs = 0; wts = nothing)
    ∂Fs = ∂cgfs(layer, inputs)
    ∂F = wmean(∂Fs; wts, dims = (ndims(layer.par) + 1):ndims(∂Fs))
    return reshape(∂F, size(layer.par))
end
