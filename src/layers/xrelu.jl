@doc raw"""
    xReLU(; θ, γ, Δ, ξ)

Extended ReLU layer, like pReLU but with unbounded asymmetry parameter.
The energy of a layer with units ``h_\mu`` is ``E = \sum_\mu U(h_\mu)``, with the
unit potential:

```math
U(h) = \begin{cases}
    \frac{|\gamma|}{2(1+\eta)} h^2 - \left(\theta + \frac{\Delta}{1+\eta}\right) h & h \ge 0 \\[4pt]
    \frac{|\gamma|}{2(1-\eta)} h^2 - \left(\theta - \frac{\Delta}{1-\eta}\right) h & h < 0
\end{cases}
\qquad \eta = \frac{\xi}{1 + |\xi|}
```

where ``\theta, \gamma, \Delta, \xi`` are the entries of `θ`, `γ`, `Δ`, `ξ` for
the corresponding unit. This is the `pReLU` potential with the bounded asymmetry
``\eta \in (-1, 1)`` reparameterized through the unbounded ``\xi \in \mathbb{R}``.
"""
@declare_layer xReLU (θ = zeros, γ = ones, Δ = zeros, ξ = zeros)

function ∂energy_from_moments(layer::xReLU, moments::AbstractArray)
    @assert ntuple(d -> size(moments, d), ndims(layer.par)) == size(layer.par)

    xp1 = @view moments[1, ..]
    xn1 = @view moments[2, ..]
    xp2 = @view moments[3, ..]
    xn2 = @view moments[4, ..]

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ, ∂γ, ∂Δ = _prelu_∂θγΔ(layer.γ, η, xp1, xn1, xp2, xn2)
    ∂ξ = @. (
        (-abs(layer.γ) / 2 * xp2 + layer.Δ * xp1) / (1 + layer.ξ + abs(layer.ξ))^2 +
            (abs(layer.γ) / 2 * xn2 + layer.Δ * xn1) / (1 - layer.ξ + abs(layer.ξ))^2
    )

    return stack([∂θ, ∂γ, ∂Δ, ∂ξ]; dims = 1)
end
