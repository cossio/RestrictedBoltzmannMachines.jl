"""
    update!(∂, x, optim)

Computes parameter update step according to `optim` algorithm (e.g. ADAM), and gradient `∂`
of parameters `x`.
Overwrites `∂` with update step and returns it.
Note that this does not update parameters.
"""
update!(∂::AbstractArray, x::AbstractArray, optim) = ∂ .= Flux.Optimise.apply!(optim, x, ∂)
function update!(∂::NamedTuple, x::Union{AbstractRBM,AbstractLayer}, optim)
    for (k, g) in pairs(∂)
        if hasproperty(x, k)
            update!(g, getproperty(x, k), optim)
        else
            g .= 0
        end
    end
    return ∂
end

"""
    update!(x, ∂)

Updates parameters according to steps `∂`.
"""
update!(x::AbstractArray, ∂::AbstractArray) = x .-= ∂
function update!(x::Union{AbstractRBM, AbstractLayer}, ∂::NamedTuple)
    for (k, Δ) in pairs(∂)
        hasproperty(x, k) && update!(getproperty(x, k), Δ)
    end
    return x
end

"""
    gradnorms(∂)

Computes gradient norms.
"""
gradnorms(∂::AbstractArray) = norm(∂)
gradnorms(∂::NamedTuple) = map(gradnorms, ∂)

"""
    default_optimizer(nsamples, batchsize, epochs; optim = ADAM(), decay_after = 0.5)

The default optimizer decays the learning rate exponentially every epoch, starting after
`decay_after` of training time, with a pre-defined schedule.
Based on defaults from https://github.com/jertubiana/PGM.
"""
function default_optimizer(
    nsamples::Int, batchsize::Int, epochs::Int;
    decay_final::Real = 1e-2, decay_after::Real = 0.5, clip = 1,
    optim = ADAM(5e-3, (0.99, 0.99), 1e-3)
)
    steps_per_epoch = minibatch_count(nsamples; batchsize = batchsize)
    nsteps = steps_per_epoch * epochs
    start = round(Int, nsteps * decay_after)
    decay = decay_final^inv(count((steps_per_epoch:steps_per_epoch:nsteps) .> start))
    return Flux.Optimise.Optimiser(
        Flux.ClipValue(clip), optim, Flux.ExpDecay(1, decay, steps_per_epoch, decay_final, start)
    )
end
