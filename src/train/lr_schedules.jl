export SqrtDecay, GeometricDecay

#= Learning rate decay schedules. Based on the implementations in Flux. =#

"""
    SqrtDecay

Learning rate decay of the form 1/sqrt(iter).
"""
mutable struct SqrtDecay
    lr0::Float64
    lrmin::Float64
    invdecay::Float64
    iter::IdDict
end

SqrtDecay(; lr0=1, lrmin=0, invdecay=0.7) = SqrtDecay(lr0, lrmin, invdecay, IdDict())

function Flux.Optimise.apply!(o::SqrtDecay, x, Δ)
    t::Int = get!(o.iter, x, 0)
    lr_t = o.lr0 / √(1 + t/o.invdecay)
    lr = max(lr_t, o.lrmin)
    Δ .*= lr
    o.iter[x] += 1
    return Δ
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
    lr_t::IdDict
end

GeometricDecay(; lr0=1, lrmin=0, decay=0.9999) = GeometricDecay(lr0, lrmin, decay, IdDict())

function apply!(o::GeometricDecay, x, Δ)
    lr_t::Float64 = get!(o.lr_t, x, o.lr0)
    lr = max(lr_t, o.lrmin)
    Δ .*= lr
    o.lr_t[x] *= o.decay
    return Δ
end
