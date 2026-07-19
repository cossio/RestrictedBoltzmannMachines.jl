using Test: @test, @testset
using Statistics: mean, std
using RestrictedBoltzmannMachines: RBM, Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, dReLU, pReLU, xReLU,
    mean_from_inputs, std_from_inputs, var_from_inputs, initialize!, onehot_decode, onehot_encode

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

@testset "initialize! layers without data" begin
    layers = [
        Binary((3,)), Spin((3,)), Potts((3, 2)), PottsGumbel((3, 2)),
        Gaussian((3,)), ReLU((3,)), dReLU((3,)), pReLU((3,)), xReLU((3,)),
    ]
    for layer in layers
        initialize!(layer)
        if layer isa dReLU
            @test all(iszero, layer.θp)
            @test all(iszero, layer.θn)
            @test all(isone, layer.γp)
            @test all(isone, layer.γn)
        else
            @test all(iszero, layer.θ)
            if layer isa Union{Gaussian, ReLU}
                @test all(isone, layer.γ)
            elseif layer isa pReLU
                @test all(isone, layer.γ)
                @test all(iszero, layer.Δ)
                @test all(iszero, layer.η)
            elseif layer isa xReLU
                @test all(isone, layer.γ)
                @test all(iszero, layer.Δ)
                @test all(iszero, layer.ξ)
            end
        end
    end
end

@testset "initialize! RBM without data" begin
    rbm = RBM(Binary((5,)), Binary((3,)), zeros(5, 3))
    @test initialize!(rbm) === rbm
    @test all(iszero, rbm.visible.θ)
    @test all(iszero, rbm.hidden.θ)
    @test !all(iszero, rbm.w)
end
