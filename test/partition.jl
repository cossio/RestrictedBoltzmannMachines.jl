include("tests_init.jl")

@testset "Gaussian-Gaussian RBM partition function (analytical)" begin
    N = (10, 3)
    M = (7, 2)
    rbm = RBMs.RBM(RBMs.Gaussian(N...), RBMs.Gaussian(M...), randn(N..., M...) / 1e3)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    randn!(rbm.visible.γ)
    randn!(rbm.hidden.γ)

    γv = vec(abs.(rbm.visible.γ))
    γh = vec(abs.(rbm.hidden.γ))
    w = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))

    A = [diagm(γv) -w;
         -w' diagm(γh)]
    θ = [vec(rbm.visible.θ); vec(rbm.hidden.θ)]

    v = randn(N..., 1)
    h = randn(M..., 1)
    x = [
        reshape(v, length(rbm.visible), 1);
        reshape(h, length(rbm.hidden),  1)
    ]

    @test RBMs.energy(rbm, v, h) ≈ x' * A * x / 2 - θ' * x
    β = rand()
    @test RBMs.log_partition(rbm, β) ≈ -logdet(β * A)/2 + β^2 * θ' * inv(β * A) * θ / 2 + (length(rbm.visible) + length(rbm.hidden)) / 2 * log(2π)
    @test RBMs.log_partition(rbm, 1) ≈ RBMs.log_partition(rbm)

    rbm.weights .= 0
    ps = Flux.params(rbm)
    gs = Zygote.gradient(ps) do
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
            Z += exp(-β * only(RBMs.energy(rbm, v, h)))
        end
    end
    @test log(Z) ≈ RBMs.log_partition(rbm, β)

    rbm.weights .= 0
    ps = Flux.params(rbm)
    gs = Zygote.gradient(ps) do
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
            Z += exp(-β * only(RBMs.energy(rbm, v, h)))
        end
    end
    @test log(Z) ≈ RBMs.log_partition(rbm, β)

    rbm.weights .= 0
    ps = Flux.params(rbm)
    gs = Zygote.gradient(ps) do
        RBMs.log_partition(rbm)
    end
    @test gs[rbm.visible.θ] ≈ tanh.(rbm.visible.θ)
    @test gs[rbm.hidden.θ]  ≈ LogExpFunctions.log1pexp.(rbm.hidden.θ)
end
