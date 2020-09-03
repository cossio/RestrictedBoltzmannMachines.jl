using Test, Random, Flux, RestrictedBoltzmannMachines
using Flux: Optimise, Optimiser, Params

@testset "Learning rate decays" begin
    Random.seed!(84)
    w = randn(10, 10)
    @testset for Opt in [SqrtDecay, GeometricDecay]
        Random.seed!(42)
        w′ = randn(10, 10)
        loss(x) = Flux.Losses.mse(w*x, w′*x)
        opt = Optimiser(Opt(), ADAM(0.001))
        for t = 1:10^5
            θ = Params([w′])
            x = rand(10)
            θ̄ = gradient(() -> loss(x), θ)
            Optimise.update!(opt, θ, θ̄)
        end
        @test loss(rand(10, 10)) < 0.01
    end
end
