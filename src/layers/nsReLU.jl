"""
    nsReLU

A variant of `xReLU` units without scale parameter γ (which is fixed at 1). This is done
to remove the gauge invariance between the weights and the hidden units scale.
"""
@declare_layer nsReLU (θ = zeros, Δ = zeros, ξ = zeros) # there is no γ

energies(layer::nsReLU, x::AbstractArray) = energies(xReLU(layer), x)
cgfs(layer::nsReLU, inputs = 0) = cgfs(xReLU(layer), inputs)
sample_from_inputs(layer::nsReLU, inputs = 0) = sample_from_inputs(xReLU(layer), inputs)
mode_from_inputs(layer::nsReLU, inputs = 0) = mode_from_inputs(xReLU(layer), inputs)
mean_from_inputs(layer::nsReLU, inputs = 0) = mean_from_inputs(xReLU(layer), inputs)
var_from_inputs(layer::nsReLU, inputs = 0) = var_from_inputs(xReLU(layer), inputs)
meanvar_from_inputs(layer::nsReLU, inputs = 0) = meanvar_from_inputs(xReLU(layer), inputs)
mean_abs_from_inputs(layer::nsReLU, inputs = 0) = mean_abs_from_inputs(xReLU(layer), inputs)

function ∂cgfs(layer::nsReLU, inputs = 0)
    xrelu = xReLU(layer)
    ∂ = ∂cgfs(xrelu, inputs)
    return ∂[[1, 3, 4], ..] # skip γ
end

function ∂energy_from_moments(layer::nsReLU, moments::AbstractArray)
    @assert size(moments) == (4, size(layer)...)
    ∂ = ∂energy_from_moments(xReLU(layer), moments)
    return ∂[[1, 3, 4], ..] # skip γ
end

xReLU(layer::nsReLU) = xReLU(; layer.θ, γ = one.(layer.θ), layer.Δ, layer.ξ)
dReLU(layer::nsReLU) = dReLU(xReLU(layer))
