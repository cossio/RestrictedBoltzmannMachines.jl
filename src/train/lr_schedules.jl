export SqrtDecay

#= Learning rate decay schedules. Based on the implementations in Flux. =#

"""
    SqrtDecay

Learning rate decay of the form 1/sqrt(iter).
"""
mutable struct SqrtDecay
    lr0::Float64
    lrmin::Float64
    decay::Float64
    step::Int
    iter::IdDict
end

SqrtDecay(lr0 = 1, lrmin = 0, decay = 0.7, step = 1) = SqrtDecay(lr0, lrmin, decay, step, IdDict())

function Flux.Optimise.apply!(o::SqrtDecay, x, Δ)
    t = get!(o.iter, x, 1)
    η = max(o.lr0 / sqrt(t / o.decay), o.lrmin)
    Δ .*= η
    o.iter[x] += o.step   
    return Δ
end