include("tests_init.jl")

@testset "Gaussian-Gaussian RBM partition function (analytical)" begin
    rbm = RBMs.RBM(RBMs.Gaussian(10), RBMs.Gaussian(7), randn(10,7) / 1e3)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    randn!(rbm.visible.γ)
    randn!(rbm.hidden.γ)

    A = [diagm(abs.(rbm.visible.γ)) -rbm.weights;
         -rbm.weights' diagm(abs.(rbm.hidden.γ))]
    v = randn(length(rbm.visible))
    h = randn(length(rbm.hidden))

    @test energy(rbm, v, h) ≈ [v' h'] * A * [v; h] / 2 - rbm.visible.θ' * v - rbm.hidden.θ' * h
    @test RBMs.log_partition(rbm, 1) ≈ -logdet(1A)/2 + (length(rbm.visible) + length(rbm.hidden)) / 2 * log(2π)
    @test RBMs.log_partition(rbm, 2) ≈ -logdet(2A)/2 + (length(rbm.visible) + length(rbm.hidden)) / 2 * log(2π)
    @test RBMs.log_partition(rbm) ≈ RBMs.log_partition(rbm, 1)

    rbm.weights .= 0
    ps = Flux.params(rbm)
    gs = gradient(ps) do
        RBMs.log_partition(rbm)
    end
    @test gs[rbm.visible.θ] ≈ rbm.visible.θ ./ rbm.visible.γ
    @test gs[rbm.hidden.θ]  ≈ rbm.hidden.θ  ./ rbm.hidden.γ
end

@testset "Binary-Binary RBM partition function (brute force)" begin
    rbm = RBMs.RBM(RBMs.Binary(3), RBMs.Binary(2), randn(3,2))
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)

    β = rand()

    Z = 0.0
    for v1 in (0,1), v2 in (0,1), v3 in (0,1)
        for h1 in (0,1), h2 in (0,1)
            v = [v1;v2;v3;;]
            h = [h1;h2;;]
            Z += exp(-β * RBMs.energy(rbm, v, h))
        end
    end
    @test log(Z) ≈ RBMs.log_partition(rbm, β)

    rbm.weights .= 0
    ps = Flux.params(rbm)
    gs = gradient(ps) do
        RBMs.log_partition(rbm)
    end
    @test gs[rbm.visible.θ] ≈ LogExpFunctions.log1pexp.(rbm.visible.θ)
    @test gs[rbm.hidden.θ]  ≈ LogExpFunctions.log1pexp.(rbm.hidden.θ)
end

@testset "Spin-Binary RBM partition function (brute force)" begin
    rbm = RBMs.RBM(RBMs.Spin(3), RBMs.Binary(2), randn(3,2))
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)

    β = rand()

    Z = 0.0
    for v1 in (-1,1), v2 in (-1,1), v3 in (-1,1)
        for h1 in (0,1), h2 in (0,1)
            v = [v1;v2;v3;;]
            h = [h1;h2;;]
            Z += exp(-β * RBMs.energy(rbm, v, h))
        end
    end
    @test log(Z) ≈ RBMs.log_partition(rbm, β)

    rbm.weights .= 0
    ps = Flux.params(rbm)
    gs = gradient(ps) do
        RBMs.log_partition(rbm)
    end
    @test gs[rbm.visible.θ] ≈ tanh.(rbm.visible.θ)
    @test gs[rbm.hidden.θ]  ≈ LogExpFunctions.log1pexp.(rbm.hidden.θ)
end
