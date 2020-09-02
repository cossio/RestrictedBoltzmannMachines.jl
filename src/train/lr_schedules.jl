export SqrtDecay

#= Learning rate decay schedules. Based on the implementations in Flux. =#

"""
    SqrtDecay

Learning rate decay of the form 1/sqrt(iter).
"""
mutable struct SqrtDecay
    lrmin::Float64
    decay::Float64
    batchsize::Float64
    iter::IdDict
end

SqrtDecay(lrmin::Real = 0, decay::Real = 0.7, batchsize::Real = 1) =
    SqrtDecay(lrmin, decay, batchsize, IdDict())

function Flux.Optimise.apply!(o::SqrtDecay, x, Δ)
    t::Float64 = get!(o.iter, x, 0.0)
    Δ .*= max(sqrt(inv(1 + t/o.decay)), o.lrmin)
    o.iter[x] += o.batchsize
    return Δ
end