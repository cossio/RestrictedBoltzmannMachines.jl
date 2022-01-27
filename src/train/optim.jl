# Based on Flux.ExpDecay, but allowing a `start` parameter
mutable struct ExpDecay <: Flux.Optimise.AbstractOptimiser
    eta::Float64
    decay::Float64
    step::Int64
    clip::Float64
    start::Int64
    current::IdDict
end
function ExpDecay(eta = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4, start = 0)
    return ExpDecay(eta, decay, decay_step, clip, start, IdDict())
end
function Flux.Optimise.apply!(o::ExpDecay, x, Δ)
    s, start = o.step, o.start
    n = o.current[x] = get(o.current, x, 0) + 1
    if n > start && n % s == 0 && count(x -> x > start && x % s == 0, values(o.current)) == 1
        o.eta = max(o.eta * o.decay, o.clip)
    end
    return Δ .*= o.eta
end

mutable struct SqrtDecay <: Flux.Optimise.AbstractOptimiser
    eta::Float64
    gamma::Float64
    step::Int64
    clip::Float64
    start::Int64
    state::IdDict
end

function SqrtDecay(η = 1, γ = 0.001, step = 1000, clip = 1e-4, start = 0)
    return SqrtDecay(η, γ, step, clip, start, IdDict())
end

function Flux.Optimise.apply!(o::SqrtDecay, x, Δ)
    s, start = o.step, o.start
    γ = o.gamma
    n = get!(o.state, x, 1)
    if n > start && n % s == 0 && count(x -> x > start && x % s == 0, values(o.state)) == 1
        o.eta = max(o.eta / (1 + γ * (n - start)), o.clip)
    end
    o.state[x] = n + 1
    return Δ .*= o.eta
end

# Based on Flux.ADAM, but taking `ϵ` as parameter
mutable struct ADAM <: Flux.Optimise.AbstractOptimiser
    eta::Float64
    beta::Tuple{Float64,Float64}
    ϵ::Float64
    state::IdDict
end
ADAM(η = 0.001, β = (0.9, 0.999), ϵ = 1e-8) = ADAM(η, β, ϵ, IdDict())
function Flux.Optimise.apply!(o::ADAM, x, Δ)
    η, β = o.eta, o.beta

    mt, vt, βp = get!(o.state, x) do
        (zero(x), zero(x), Float64[β[1], β[2]])
    end :: Tuple{typeof(x),typeof(x),Vector{Float64}}

    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.ϵ) * η
    βp .= βp .* β

    return Δ
end

"""
    default_optimizer(nsamples, batchsize, epochs; opt = ADAM(), decay_after = 0.5)

The default optimizer decays the learning rate exponentially every epoch, starting after
`decay_after` of training time, with a pre-defined schedule.
Based on defaults from https://github.com/jertubiana/PGM.
"""
function default_optimizer(
    nsamples::Int, batchsize::Int, epochs::Int;
    decay_final::Real = 1e-2, decay_after::Real = 0.5, clip = 1,
    opt = ADAM(5e-3, (0.99, 0.99), 1e-3)
)
    steps_per_epoch = minibatch_count(nsamples; batchsize = batchsize)
    nsteps = steps_per_epoch * epochs
    start = round(Int, nsteps * decay_after)
    decay = decay_final^inv(count((steps_per_epoch:steps_per_epoch:nsteps) .> start))
    return Flux.Optimise.Optimiser(
        Flux.ClipValue(clip), opt, ExpDecay(1, decay, steps_per_epoch, decay_final, start)
    )
end


#=
Learning rate decay schedules. Based on the implementations in Flux. See:
https://fluxml.ai/Flux.jl/stable/training/optimisers/#Composing-Optimisers-1
https://github.com/FluxML/Flux.jl/blob/2b1ba184d1a58c37543f4561413cddb2de594289/src/optimise/optimisers.jl#L517-L538
=#

# abstract type AbstractLrDecay end

# function Flux.Optimise.apply!(o::AbstractLrDecay, x, Δ)
#     t::Int = o.t[x] = get(o.t, x, 0) + 1
#     Δ .*= update_lr(o, t)
# end

# """
#     SqrtDecay

# Learning rate decay of the form 1/sqrt(iter).
# """
# mutable struct SqrtDecay <: AbstractLrDecay
#     lr0::Float64
#     lrmin::Float64
#     decay::Float64
#     t::IdDict
# end
# SqrtDecay(; lr0=1, lrmin=0, decay=0) = SqrtDecay(lr0, lrmin, decay, IdDict())
# update_lr(o::SqrtDecay, t::Real) = max(o.lr0 / sqrt(1 + (t - 1) * o.decay), o.lrmin)
