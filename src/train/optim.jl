struct ∂RBM{V,H,W}
    visible::V
    hidden::H
    w::W
    function ∂RBM(visible::AbstractArray, hidden::AbstractArray, w::AbstractArray)
        @assert size(w) == (tail(size(visible))..., tail(size(hidden))...)
        return new{typeof(visible), typeof(hidden), typeof(w)}(visible, hidden, w)
    end
end

"""
    update!(∂rbm, rbm, optim)

Computes parameter update step according to `optim` algorithm (e.g. Adam),
and gradient `∂rbm` of parameters in `rbm`.
Overwrites `∂rbm` with update step and returns it.
Note that this does not update parameters.
"""
function update!(∂::∂RBM, rbm::RBM, optim)
    Flux.Optimise.apply!(optim, rbm.visible.par, ∂.visible)
    Flux.Optimise.apply!(optim, rbm.hidden.par, ∂.hidden)
    Flux.Optimise.apply!(optim, rbm.w, ∂.w)
    return ∂
end

"""
    update!(rbm, ∂rbm)

Updates parameters of `rbm` according to steps `∂rbm`.
"""
function update!(rbm::RBM, ∂::∂RBM)
    rbm.visible.par .= rbm.visible.par - ∂.visible
    rbm.hidden.par .= rbm.hidden.par - ∂.hidden
    rbm.w .= rbm.w - ∂.w
    return rbm
end

"""
    gradnorms(∂)

Computes gradient norms.
"""
gradnorms(∂::∂RBM) = (visible = norm(∂.visible), hidden = norm(∂.hidden), w = norm(∂.w))

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
    optim = Adam(5e-3, (0, 0.99), 1e-3)
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

Base.:(+)(∂1::∂RBM, ∂2::∂RBM) = ∂RBM(∂1.visible + ∂2.visible, ∂1.hidden + ∂2.hidden, ∂1.w + ∂2.w)
Base.:(-)(∂1::∂RBM, ∂2::∂RBM) = ∂RBM(∂1.visible - ∂2.visible, ∂1.hidden - ∂2.hidden, ∂1.w - ∂2.w)
Base.:(*)(λ::Real, ∂::∂RBM) = ∂RBM(λ * ∂.visible, λ * ∂.hidden, λ * ∂.w)
Base.:(*)(∂::∂RBM, λ::Real) = λ * ∂
Base.:(==)(∂1::∂RBM, ∂2::∂RBM) = (∂1.visible == ∂2.visible) && (∂1.hidden == ∂2.hidden) && (∂1.w == ∂2.w)
Base.hash(∂::∂RBM, h::UInt) = hash(∂.visible, hash(∂.hidden, hash(∂.w, h)))
