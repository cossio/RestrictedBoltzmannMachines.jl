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


mutable struct WaitExpDecay <: Flux.Optimise.AbstractOptimiser
    eta::Float64
    decay::Float64
    step::Int64
    clip::Float64
    from::Int64
    current::IdDict
end

function WaitExpDecay(eta = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4, from = decay_step)
    return WaitExpDecay(eta, decay, decay_step, clip, from, IdDict())
end

function apply!(o::WaitExpDecay, x, Δ)
    η, s, decay, f = o.eta, o.step, o.decay, o.from
    n = o.current[x] = get(o.current, x, 0) + 1
    if n ≥ f && (n - f) % s == 0 && count(x -> (x - f) % s == 0, values(o.current)) == 1
        η = max(η * decay, o.clip)
        o.eta = η
    end
    @. Δ *= η
end
