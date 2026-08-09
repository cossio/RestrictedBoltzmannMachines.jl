@doc raw"""
    nsReLU(; θ, Δ, ξ)

A variant of `xReLU` units without scale parameter γ (which is fixed at 1). This is done
to remove the gauge invariance between the weights and the hidden units scale.
The energy of a layer with units ``h_\mu`` is ``E = \sum_\mu U(h_\mu)``, with the
unit potential:

```math
U(h) = \begin{cases}
    \frac{1}{2(1+\eta)} h^2 - \left(\theta + \frac{\Delta}{1+\eta}\right) h & h \ge 0 \\[4pt]
    \frac{1}{2(1-\eta)} h^2 - \left(\theta - \frac{\Delta}{1-\eta}\right) h & h < 0
\end{cases}
\qquad \eta = \frac{\xi}{1 + |\xi|}
```

where ``\theta, \Delta, \xi`` are the entries of `θ`, `Δ`, `ξ` for the
corresponding unit. This is the `xReLU` potential with ``\gamma = 1``.
"""
@declare_layer nsReLU (θ = zeros, Δ = zeros, ξ = zeros) # there is no γ

energies(layer::nsReLU, x::AbstractArray) = energies(xReLU(layer), x)
cgfs(layer::nsReLU, inputs::AbstractArray = Falses(size(layer))) = cgfs(xReLU(layer), inputs)
sample_from_inputs(layer::nsReLU, inputs::AbstractArray = Falses(size(layer))) = sample_from_inputs(xReLU(layer), inputs)
mode_from_inputs(layer::nsReLU, inputs::AbstractArray = Falses(size(layer))) = mode_from_inputs(xReLU(layer), inputs)
mean_from_inputs(layer::nsReLU, inputs::AbstractArray = Falses(size(layer))) = mean_from_inputs(xReLU(layer), inputs)
var_from_inputs(layer::nsReLU, inputs::AbstractArray = Falses(size(layer))) = var_from_inputs(xReLU(layer), inputs)
meanvar_from_inputs(layer::nsReLU, inputs::AbstractArray = Falses(size(layer))) = meanvar_from_inputs(xReLU(layer), inputs)
mean_abs_from_inputs(layer::nsReLU, inputs::AbstractArray = Falses(size(layer))) = mean_abs_from_inputs(xReLU(layer), inputs)

function ∂energy_from_moments(layer::nsReLU, moments::AbstractArray)
    @assert ntuple(d -> size(moments, d), ndims(layer) + 1) == (4, size(layer)...)
    ∂ = ∂energy_from_moments(xReLU(layer), moments)
    return ∂[[1, 3, 4], ..] # skip γ
end

xReLU(layer::nsReLU) = xReLU(; layer.θ, γ = one.(layer.θ), layer.Δ, layer.ξ)
dReLU(layer::nsReLU) = dReLU(xReLU(layer))
