using Test: @test, @testset
import Random
import Zygote
import Flux
import RestrictedBoltzmannMachines as RBMs

@testset "Binary-Binary RBM partition function (brute force)" begin
    rbm = RBMs.RBM(RBMs.Binary(3), RBMs.Binary(2), randn(3,2))
    Random.randn!(rbm.visible.θ)
    Random.randn!(rbm.hidden.θ)

    β = rand()

    Z = 0.0
    for v1 in (0,1), v2 in (0,1), v3 in (0,1)
        for h1 in (0,1), h2 in (0,1)
            v = [v1;v2;v3;;]
            h = [h1;h2;;]
            Z += exp(-β * only(RBMs.energy(rbm, v, h)))
        end
    end
    @test log(Z) ≈ RBMs.log_partition(rbm; β)

    rbm.w .= 0
    gs = Zygote.gradient(rbm) do rbm
        RBMs.log_partition(rbm)
    end
    @test only(gs).visible.θ ≈ Flux.sigmoid.(rbm.visible.θ)
    @test only(gs).hidden.θ  ≈ Flux.sigmoid.(rbm.hidden.θ)
end

@testset "Spin-Binary RBM partition function (brute force)" begin
    rbm = RBMs.RBM(RBMs.Spin(3), RBMs.Binary(2), randn(3,2))
    Random.randn!(rbm.visible.θ)
    Random.randn!(rbm.hidden.θ)

    β = rand()

    Z = 0.0
    for v1 in (-1,1), v2 in (-1,1), v3 in (-1,1)
        for h1 in (0,1), h2 in (0,1)
            v = [v1;v2;v3;;]
            h = [h1;h2;;]
            Z += exp(-β * only(RBMs.energy(rbm, v, h)))
        end
    end
    @test log(Z) ≈ RBMs.log_partition(rbm; β)

    rbm.w .= 0
    gs = Zygote.gradient(rbm) do rbm
        RBMs.log_partition(rbm)
    end
    @test only(gs).visible.θ ≈ tanh.(rbm.visible.θ)
    @test only(gs).hidden.θ ≈ Flux.sigmoid.(rbm.hidden.θ)
end
