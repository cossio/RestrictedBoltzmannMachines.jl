
"""
    ExpDecay(η = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4, start = 1)

Discount the learning rate `η` by the factor `decay` every `decay_step` steps till
a minimum of `clip`.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- `decay`: Factor by which the learning rate is discounted.
- `decay_step`: Schedule decay operations by setting the number of steps between
                two decay operations.
- `clip`: Minimum value of learning rate.
- 'start': Step at which the decay starts.


See also the [Scheduling Optimisers](@ref) section of the docs
for more general scheduling techniques.

# Examples

`ExpDecay` is typically composed  with other optimizers
as the last transformation of the gradient:
```julia
opt = Optimiser(ADAM(), ExpDecay())
```
"""
mutable struct ExpDecay <: Flux.Optimise.AbstractOptimiser
    eta::Float64
    decay::Float64
    step::Int64
    clip::Float64
    start::Int64
    current::IdDict
end

function ExpDecay(opt = 0.001, decay = 0.1, decay_step = 1000, clip = 1e-4, start = 1)
    return ExpDecay(opt, decay, decay_step, clip, start, IdDict())
end

function Flux.Optimise.apply!(o::ExpDecay, x, Δ)
    η, s, decay, n0 = o.eta, o.step, o.decay, o.start
    n = o.current[x] = get(o.current, x, 0) + 1
    if n ≥ n0 && (n - n0 + 1) % s == 0 && count(x -> (x - n0 + 1) % s == 0, values(o.current)) == 1
        η = max(η * decay, o.clip)
        o.eta = η
    end
    @. Δ *= η
end