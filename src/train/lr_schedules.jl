#=
Learning rate decay schedules. Based on the implementations in Flux. See:
https://fluxml.ai/Flux.jl/stable/training/optimisers/#Composing-Optimisers-1
https://github.com/FluxML/Flux.jl/blob/2b1ba184d1a58c37543f4561413cddb2de594289/src/optimise/optimisers.jl#L517-L538
=#

import Flux

abstract type AbstractLrDecay end

function Flux.Optimise.apply!(o::AbstractLrDecay, x, Δ)
    t::Int = o.t[x] = get(o.t, x, 0) + 1
    Δ .*= update_lr(o, t)
end

"""
    SqrtDecay

Learning rate decay of the form 1/sqrt(iter).
"""
mutable struct SqrtDecay <: AbstractLrDecay
    lr0::Float64
    lrmin::Float64
    decay::Float64
    t::IdDict
end
SqrtDecay(; lr0=1, lrmin=0, decay=0) = SqrtDecay(lr0, lrmin, decay, IdDict())
update_lr(o::SqrtDecay, t::Real) = max(o.lr0 / sqrt(1 + (t - 1) * o.decay), o.lrmin)
