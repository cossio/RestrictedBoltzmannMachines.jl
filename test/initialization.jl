using Test: @test, @testset
using Statistics: mean, std
using Random: randn!
using RestrictedBoltzmannMachines: RBM, Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, dReLU, pReLU, xReLU, nsReLU, mean_from_inputs, std_from_inputs, initialize!, onehot_decode, onehot_encode

@testset "initialization Binary" begin
    data = rand(2, 3, 10^6) .≤ 3 / 4
    rbm = RBM(Binary((2, 3)), Binary((0,)), randn(2, 3, 0))
    initialize!(rbm, data)
    @test mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims = 3), size(rbm.visible)) rtol = 0.01
end

@testset "initialization Spin" begin
    data = Int8.(sign.(rand(2, 3, 10^6) .- 1 / 4))
    rbm = RBM(Spin((2, 3)), Binary((0,)), randn(2, 3, 0))
    initialize!(rbm, data)
    @test mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims = 3), size(rbm.visible)) rtol = 0.01
end

@testset "initialization Potts" begin
    rbm = RBM(Potts((3, 2, 3)), Binary((0,)), randn(3, 2, 3, 0))
    data = onehot_encode(onehot_decode(rand(3, 2, 3, 10^6) .- rand(3, 2, 3, 1)), 1:3)
    @assert all(sum(data; dims = 1) .== 1)
    initialize!(rbm, data)
    @test mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims = 4), size(rbm.visible)) rtol = 0.01
end

@testset "initialization Gaussian" begin
    data = 0.5randn(5, 10^6) .+ 1
    rbm = RBM(Gaussian((5,)), Binary((0,)), randn(5, 0))
    initialize!(rbm, data)
    @test mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims = 2), size(rbm.visible)) rtol = 0.01
    @test std_from_inputs(rbm.visible) ≈ reshape(std(data; dims = 2), size(rbm.visible)) rtol = 0.01
end

@testset "initialization dReLU" begin
    data = 0.5randn(5, 10^6) .+ 1
    rbm = RBM(dReLU((5,)), Binary((0,)), randn(5, 0))
    initialize!(rbm, data)
    @test mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims = 2), size(rbm.visible)) rtol = 0.01
    @test std_from_inputs(rbm.visible) ≈ reshape(std(data; dims = 2), size(rbm.visible)) rtol = 0.01
end

@testset "initialization pReLU" begin
    data = 0.5randn(5, 10^6) .+ 1
    rbm = RBM(pReLU((5,)), Binary((0,)), randn(5, 0))
    initialize!(rbm, data)
    @test mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims = 2), size(rbm.visible)) rtol = 0.01
    @test std_from_inputs(rbm.visible) ≈ reshape(std(data; dims = 2), size(rbm.visible)) rtol = 0.01
end

@testset "initialization xReLU" begin
    data = 0.5randn(5, 10^6) .+ 1
    rbm = RBM(xReLU((5,)), Binary((0,)), randn(5, 0))
    initialize!(rbm, data)
    @test mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims = 2), size(rbm.visible)) rtol = 0.01
    @test std_from_inputs(rbm.visible) ≈ reshape(std(data; dims = 2), size(rbm.visible)) rtol = 0.01
end

@testset "initialization nsReLU" begin
    data = [1.0 3.0 5.0; -2.0 0.0 4.0]
    for (kwargs, μ) in (
            ((;), [3, 2 / 3]),
            ((; ϵ = 0.1, wts = [1.0, 2.0, 3.0]), [11 / 3, 5 / 3]),
        )
        rbm = RBM(nsReLU((2,)), Binary((1,)), zeros(2, 1))
        rbm.visible.par .= 1
        rbm.hidden.par .= 1
        @test initialize!(rbm, data; kwargs...) === rbm
        @test rbm.visible.θ ≈ μ
        @test all(iszero, rbm.visible.Δ)
        @test all(iszero, rbm.visible.ξ)
        @test mean_from_inputs(rbm.visible) ≈ μ
        @test std_from_inputs(rbm.visible) ≈ ones(2)
        @test all(iszero, rbm.hidden.θ)
        @test all(isfinite, rbm.w)
        @test !all(iszero, rbm.w)
    end
end

@testset "initialize! layers without data" begin
    for Layer in (Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, dReLU, pReLU, xReLU, nsReLU)
        layer = Layer((3, 2))
        randn!(layer.par)
        @test initialize!(layer) === layer
        @test layer.par == Layer((3, 2)).par
    end
end

@testset "initialize! RBM without data" begin
    rbm = RBM(Binary((5,)), Binary((3,)), zeros(5, 3))
    @test initialize!(rbm) === rbm
    @test all(iszero, rbm.visible.θ)
    @test all(iszero, rbm.hidden.θ)
    @test !all(iszero, rbm.w)
end
