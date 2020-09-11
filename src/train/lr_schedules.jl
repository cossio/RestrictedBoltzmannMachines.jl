#=
Learning rate decay schedules. Based on the implementations in Flux. See:
https://fluxml.ai/Flux.jl/stable/training/optimisers/#Composing-Optimisers-1
https://github.com/FluxML/Flux.jl/blob/2b1ba184d1a58c37543f4561413cddb2de594289/src/optimise/optimisers.jl#L517-L538
=#

using Flux: Optimiser
export Optimiser, SqrtDecay, GeometricDecay

"""
    SqrtDecay

Learning rate decay of the form 1/sqrt(iter).
"""
mutable struct SqrtDecay
    lr0::Float64
    lrmin::Float64
    decay::Float64
    t::IdDict
end

SqrtDecay(; lr0=1, lrmin=0, decay=1) = SqrtDecay(lr0, lrmin, decay, IdDict())

function Flux.Optimise.apply!(o::SqrtDecay, x, Δ)
    t::Int = o.t[x] = get(o.t, x, 0) + 1
    Δ .*= max(o.lr0 / √(1 + (t - 1) * o.decay), o.lrmin)
end


"""
    GeometricDecay

Geometric (a.k.a. exponential) decay of learning rate.
Similar to Flux.Optimisers.ExpDecay, but with a slightly
different implementation.
"""
mutable struct GeometricDecay
    lr0::Float64
    lrmin::Float64
    decay::Float64
    t::IdDict
end

GeometricDecay(; lr0=1, lrmin=0, decay=0.9999) = GeometricDecay(lr0, lrmin, decay, IdDict())

function Flux.Optimise.apply!(o::GeometricDecay, x, Δ)
    t::Int = o.t[x] = get(o.t, x, 0) + 1
    Δ .*= max(o.lr0 * o.decay^t, o.lrmin)
end
