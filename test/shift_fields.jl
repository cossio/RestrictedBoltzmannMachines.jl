using Test: @testset, @test
using RestrictedBoltzmannMachines: Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU,
    energy, sample_from_inputs, shift_fields!, shift_fields

function energy_shift(offset::AbstractArray, x::AbstractArray)
    @assert size(offset) == size(x)[1:ndims(offset)]
    if ndims(offset) == ndims(x)
        return -sum(offset .* x)
    elseif ndims(offset) < ndims(x)
        ΔE = -sum(offset .* x; dims=1:ndims(offset))
        return reshape(ΔE, size(x)[(ndims(offset) + 1):end])
    end
end

@testset "shift_fields" begin
    N = (3, 4)
    layers = (
        Binary(; θ = randn(N...)),
        Spin(; θ = randn(N...)),
        Potts(; θ = randn(N...)),
        Gaussian(; θ = randn(N...), γ = rand(N...)),
        ReLU(; θ = randn(N...), γ = rand(N...)),
        dReLU(; θp = randn(N...), θn = randn(N...), γp = rand(N...), γn = rand(N...)),
        pReLU(; θ = randn(N...), γ = rand(N...), Δ = randn(N...), η = rand(N...) .- 0.5),
        xReLU(; θ = randn(N...), γ = rand(N...), Δ = randn(N...), ξ = randn(N...)),
    )
    for layer in layers
        offset = randn(size(layer)...)
        x = sample_from_inputs(layer, randn(size(layer)..., 2, 3))
        layer_shifted = @inferred shift_fields(layer, offset)
        @test energy(layer_shifted, x) ≈ energy(layer, x) + energy_shift(offset, x)
    end
end

@testset "shift_fields!" begin
    N = (3, 4)
    layers = (
        Binary(; θ = randn(N...)),
        Spin(; θ = randn(N...)),
        Potts(; θ = randn(N...)),
        Gaussian(; θ = randn(N...), γ = rand(N...)),
        ReLU(; θ = randn(N...), γ = rand(N...)),
        dReLU(; θp = randn(N...), θn = randn(N...), γp = rand(N...), γn = rand(N...)),
        pReLU(; θ = randn(N...), γ = rand(N...), Δ = randn(N...), η = rand(N...) .- 0.5),
        xReLU(; θ = randn(N...), γ = rand(N...), Δ = randn(N...), ξ = randn(N...)),
    )
    for layer in layers
        offset = randn(size(layer)...)
        x = sample_from_inputs(layer, randn(size(layer)..., 2, 3))
        layer_shifted = @inferred shift_fields!(deepcopy(layer), offset)
        @test energy(layer_shifted, x) ≈ energy(layer, x) + energy_shift(offset, x)
    end
end
