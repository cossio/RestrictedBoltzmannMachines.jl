export SqrtDecay

#= Learning rate decay schedules. Based on the implementations in Flux. =#

"""
    SqrtDecay

Learning rate decay of the form 1/sqrt(iter).
"""
mutable struct SqrtDecay
    lr0::Float64
    lrmin::Float64
    invdecay::Float64
    batchsize::Float64
    iter::IdDict
end

SqrtDecay(; lr0::Real = 1, lrmin::Real = 0, invdecay::Real = 0.7, batchsize::Real = 1) =
    SqrtDecay(lr0, lrmin, invdecay, batchsize, IdDict())

function Flux.Optimise.apply!(o::SqrtDecay, x, Δ)
    t::Float64 = get!(o.iter, x, 0.0)
    Δ .*= max(o.lr0 / √(1 + t/o.invdecay), o.lrmin)
    o.iter[x] += o.batchsize
    return Δ
end
