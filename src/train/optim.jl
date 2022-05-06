"""
    update!(∂, x, optim)

Computes parameter update step according to `optim` algorithm (e.g. ADAM), and gradient `∂`
of parameters `x`.
Overwrites `∂` with update step and returns it.
Note that this does not update parameters.
"""
function update!(∂::AbstractArray, x::AbstractArray, optim)
    ∂ .= Flux.Optimise.apply!(optim, x, ∂)
    return ∂
end

function update!(∂::NamedTuple, rbm::RBM, optim)
    update!(∂.visible, rbm.visible, optim)
    update!(∂.hidden, rbm.hidden, optim)
    update!(∂.w, rbm.w, optim)
    return ∂
end

function update!(∂::NamedTuple, l::Union{Binary,Spin,Potts}, optim)
    update!(∂.θ, l.θ, optim)
    return ∂
end

function update!(∂::NamedTuple, l::Union{Gaussian,ReLU}, optim)
    update!(∂.θ, l.θ, optim)
    update!(∂.γ, l.γ, optim)
    return ∂
end

function update!(∂::NamedTuple, l::dReLU, optim)
    update!(∂.θp, l.θp, optim)
    update!(∂.θn, l.θn, optim)
    update!(∂.γp, l.γp, optim)
    update!(∂.γn, l.γn, optim)
    return ∂
end

function update!(∂::NamedTuple, l::pReLU, optim)
    update!(∂.θ, l.θ, optim)
    update!(∂.γ, l.γ, optim)
    update!(∂.Δ, l.Δ, optim)
    update!(∂.η, l.η, optim)
    return ∂
end

function update!(∂::NamedTuple, l::xReLU, optim)
    update!(∂.θ, l.θ, optim)
    update!(∂.γ, l.γ, optim)
    update!(∂.Δ, l.Δ, optim)
    update!(∂.ξ, l.ξ, optim)
    return ∂
end

"""
    update!(x, ∂)

Updates parameters according to steps `∂`.
"""
function update!(x::AbstractArray, ∂::AbstractArray)
    x .-= ∂
    return x
end

function update!(rbm::RBM, ∂::NamedTuple)
    update!(rbm.visible, ∂.visible)
    update!(rbm.hidden, ∂.hidden)
    update!(rbm.w, ∂.w)
    return rbm
end

function update!(l::Union{Binary,Spin,Potts}, ∂::NamedTuple)
    update!(l.θ, ∂.θ)
    return l
end

function update!(l::Union{Gaussian,ReLU}, ∂::NamedTuple)
    update!(l.θ, ∂.θ)
    update!(l.γ, ∂.γ)
    return l
end

function update!(l::dReLU, ∂::NamedTuple)
    update!(l.θp, ∂.θp)
    update!(l.θn, ∂.θn)
    update!(l.γp, ∂.γp)
    update!(l.γn, ∂.γn)
    return l
end

function update!(l::pReLU, ∂::NamedTuple)
    update!(l.θ, ∂.θ)
    update!(l.γ, ∂.γ)
    update!(l.Δ, ∂.Δ)
    update!(l.η, ∂.η)
    return l
end

function update!(l::xReLU, ∂::NamedTuple)
    update!(l.θ, ∂.θ)
    update!(l.γ, ∂.γ)
    update!(l.Δ, ∂.Δ)
    update!(l.ξ, ∂.ξ)
    return l
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
    # See also 10.1016/j.cels.2020.11.005 (search RMSprop)
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

subtract_gradients(∂1::NamedTuple, ∂2::NamedTuple) = map(subtract_gradients, ∂1, ∂2)
subtract_gradients(∂1::AbstractArray, ∂2::AbstractArray) = ∂1 - ∂2

add_gradients(∂s::NamedTuple...) = map(add_gradients, ∂s...)
add_gradients(∂s::AbstractArray...) = +(∂s...)

function combine_gradients(op, ∂1::NamedTuple, ∂2::NamedTuple)
    _op(∂1::NamedTuple, ∂2::NamedTuple) = map(_op, ∂1, ∂2)
    _op(∂1::AbstractArray, ∂2::AbstractArray) = op(∂1, ∂2)
    return map(_op, ∂1, ∂2)
end
