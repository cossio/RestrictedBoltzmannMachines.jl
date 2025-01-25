using LinearAlgebra: I
using LinearAlgebra: norm
using Random: bitrand
using RestrictedBoltzmannMachines: ∂free_energy
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: BinaryRBM
using RestrictedBoltzmannMachines: BinaryStandardizedRBM
using RestrictedBoltzmannMachines: delta_energy
using RestrictedBoltzmannMachines: dReLU
using RestrictedBoltzmannMachines: energy
using RestrictedBoltzmannMachines: free_energy
using RestrictedBoltzmannMachines: Gaussian
using RestrictedBoltzmannMachines: inputs_h_from_v
using RestrictedBoltzmannMachines: inputs_v_from_h
using RestrictedBoltzmannMachines: interaction_energy
using RestrictedBoltzmannMachines: mean_h_from_v
using RestrictedBoltzmannMachines: mean_v_from_h
using RestrictedBoltzmannMachines: pcd!
using RestrictedBoltzmannMachines: Potts
using RestrictedBoltzmannMachines: pReLU
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: ReLU
using RestrictedBoltzmannMachines: rescale_hidden_activations!
using RestrictedBoltzmannMachines: sample_from_inputs
using RestrictedBoltzmannMachines: shift_fields
using RestrictedBoltzmannMachines: shift_fields!
using RestrictedBoltzmannMachines: Spin
using RestrictedBoltzmannMachines: SpinStandardizedRBM
using RestrictedBoltzmannMachines: standardize
using RestrictedBoltzmannMachines: standardize_hidden
using RestrictedBoltzmannMachines: standardize_hidden!
using RestrictedBoltzmannMachines: standardize_visible
using RestrictedBoltzmannMachines: standardize_visible!
using RestrictedBoltzmannMachines: standardize!
using RestrictedBoltzmannMachines: StandardizedRBM
using RestrictedBoltzmannMachines: unstandardize
using RestrictedBoltzmannMachines: var_h_from_v
using RestrictedBoltzmannMachines: var_v_from_h
using RestrictedBoltzmannMachines: xReLU
using Statistics: mean
using Test: @inferred
using Test: @test
using Test: @testset
using Zygote: gradient

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
        E = energy(layer, x)
        @inferred shift_fields!(layer, offset)
        @test energy(layer, x) ≈ E + energy_shift(offset, x)
    end
end

@testset "standardize" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    offset_v = randn(3)
    offset_h = randn(2)
    scale_v = rand(3)
    scale_h = rand(2)

    std_rbm = @inferred standardize(rbm, offset_v, offset_h, scale_v, scale_h)
    @test std_rbm.offset_v ≈ offset_v
    @test std_rbm.offset_h ≈ offset_h
    @test std_rbm.scale_v ≈ scale_v
    @test std_rbm.scale_h ≈ scale_h

    @test std_rbm.w ./ (reshape(scale_v, 3, 1) .* reshape(scale_h, 1, 2)) ≈ rbm.w

    v = bitrand(3, 10)
    h = bitrand(2, 10)
    @test energy(std_rbm, v, h) .- delta_energy(std_rbm) ≈ energy(rbm, v, h)
    @test free_energy(std_rbm, v) .- delta_energy(std_rbm) ≈ free_energy(rbm, v)

    @test iszero(standardize(rbm).offset_v)
    @test iszero(standardize(rbm).offset_h)
    @test all(standardize(rbm).scale_v .== 1)
    @test all(standardize(rbm).scale_h .== 1)
    @inferred standardize(rbm)
    @test iszero(standardize(std_rbm).offset_v)
    @test iszero(standardize(std_rbm).offset_h)
    @test all(standardize(std_rbm).scale_v .== 1)
    @test all(standardize(std_rbm).scale_h .== 1)
    @inferred standardize(std_rbm)
    @test energy(rbm, v, h) ≈ energy(standardize(rbm), v, h) ≈ energy(standardize(std_rbm), v, h)
end

@testset "unstandardize" begin
    std_rbm = @inferred BinaryStandardizedRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2), rand(3), rand(2))
    rbm = @inferred unstandardize(std_rbm)
    @test rbm isa RBM
    @test unstandardize(rbm) == rbm
    v = bitrand(3, 10)
    h = bitrand(2, 10)
    @test energy(std_rbm, v, h) .- delta_energy(std_rbm) ≈ energy(rbm, v, h)
end

@testset "delta_energy" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    std_rbm = @inferred standardize(rbm, randn(3), randn(2), rand(3), rand(2))
    @test iszero(@inferred delta_energy(rbm))
    @test iszero(delta_energy(standardize(std_rbm)))
    @test delta_energy(rbm) isa Real
    @test delta_energy(std_rbm) isa Real
end

@testset "standardize!" begin
    rbm = @inferred BinaryStandardizedRBM(
        randn(3), randn(2), randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    offset_v = randn(3)
    offset_h = randn(2)
    scale_v = rand(3)
    scale_h = rand(2)

    v = bitrand(3, 10)
    h = bitrand(2, 10)
    E = energy(rbm, v, h) .- delta_energy(rbm)
    F = free_energy(rbm, v) .- delta_energy(rbm)

    standardize!(rbm, offset_v, offset_h, scale_v, scale_h)
    @test rbm.offset_v ≈ offset_v
    @test rbm.offset_h ≈ offset_h
    @test rbm.scale_v ≈ scale_v
    @test rbm.scale_h ≈ scale_h

    @test energy(rbm, v, h) .- delta_energy(rbm) ≈ E
    @test free_energy(rbm, v) .- delta_energy(rbm) ≈ F
end

@testset "∂free energy" begin
    rbm = @inferred BinaryStandardizedRBM(
        randn(3), randn(2), randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    v = bitrand(size(rbm.visible)..., 10)
    gs = gradient(rbm) do rbm
        mean(free_energy(rbm, v))
    end
    ∂ = ∂free_energy(rbm, v)
    @test ∂.visible ≈ only(gs).visible.par
    @test ∂.hidden ≈ only(gs).hidden.par
    @test ∂.w ≈ only(gs).w
end

@testset "standardized constructor" begin
    rbm = SpinStandardizedRBM(randn(10), randn(7), randn(10, 7))
    @test rbm.visible isa Spin
    @test rbm.hidden isa Spin
end

@testset "rescale_hidden_activations!" begin
    rbm = standardize(
        RBM(Binary(; θ=randn(3)), ReLU(; θ=randn(2), γ=0.1 .+ rand(2)), randn(3,2)),
        randn(3), randn(2), 0.1 .+ rand(3), 0.1 .+ rand(2)
    )

    v = bitrand(3, 1000)
    F = free_energy(rbm, v)
    h_ave = mean_h_from_v(rbm, v)
    h_var = var_h_from_v(rbm, v)
    v_ave = mean_v_from_h(rbm, h_ave)
    v_var = var_v_from_h(rbm, h_ave)

    λ = copy(rbm.scale_h)
    rescale_hidden_activations!(rbm)

    @test mean_h_from_v(rbm, v) ≈ h_ave ./ λ
    @test var_h_from_v(rbm, v) ≈ h_var ./ λ.^2
    @test mean_v_from_h(rbm, h_ave ./ λ) ≈ v_ave
    @test var_v_from_h(rbm, h_ave ./ λ) ≈ v_var
    @test all(rbm.scale_h .≈ 1)
    @test free_energy(rbm, v) ≈ F .+ sum(log, λ)
end

@testset "standardized pcd" begin
    rbm = standardize(RBM(Spin(; θ=zeros(10)), Spin(; θ=zeros(7)), randn(10, 7) / √10))
    @test iszero(rbm.visible.θ) && iszero(rbm.hidden.θ)

    data = ones(10, 4)
    data[1:3, 2] .= -1
    data[:, 3:4] .= -data[:, 1:2] # ensure data has zero mean
    @test iszero(mean(data; dims=2))

    state, ps = pcd!(
        rbm, data;
        ps = (; w=rbm.w), # train only weights
        steps=10, batchsize=4, iters=1000,
        ϵv=1f-1, ϵh=0f0, damping=1f-1
    )

    # The fields are not exaclty zero because centering introduces minor numerical fluctuations.
    @test norm(rbm.visible.θ) < 1e-13
    @test iszero(rbm.hidden.θ)
end
