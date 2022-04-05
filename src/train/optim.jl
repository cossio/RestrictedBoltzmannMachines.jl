"""
    update!(∂, x, optim)

Computes parameter update step according to `optim` algorithm (e.g. ADAM), and gradient `∂`
of parameters `x`.
Overwrites `∂` with update step and returns it.
Note that this does not update parameters.
"""
function update!(∂::AbstractArray, x::AbstractArray, optim)
    ∂ .= Flux.Optimise.apply!(optim, x, ∂)
end

function update!(∂::NamedTuple, x::Union{RBM,AbstractLayer}, optim)
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
function update!(x::Union{RBM, AbstractLayer}, ∂::NamedTuple)
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
    default_optimizer(nsamples, batchsize, epochs; decay_after, decay_final, clip, optim)

Sane defaults for optimizer. All keyword arguments are optional and have default settings.
Trains with constant learning rate for first `decay_after` period, then decays
learning rate exponentially every epoch, starting after
`decay_after` of training time, until reaching `decay_final` at the last epoch.
Clips gradients by `clip`.
"""
function default_optimizer(
    nsamples::Int, batchsize::Int, epochs::Int;
    decay_final::Real = 0.01, decay_after::Real = 0.5, clip = 1,
    optim = ADAM(5e-3, (0, 0.99), 1e-3)
)
    # Defaults from https://github.com/jertubiana/PGM
    steps_per_epoch = minibatch_count(nsamples; batchsize)
    startepoch = round(Int, epochs * decay_after)
    start = startepoch * steps_per_epoch - 1
    decay = decay_final^(1/(epochs - startepoch))
    return Flux.Optimise.Optimiser(
        Flux.ClipValue(clip),
        optim,
        Flux.ExpDecay(1, decay, steps_per_epoch, decay_final, start)
    )
end
