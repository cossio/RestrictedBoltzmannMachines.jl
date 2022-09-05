using Test: @test, @testset
using Random: randn!
import Random
import Zygote
import Flux
import RestrictedBoltzmannMachines as RBMs
using RestrictedBoltzmannMachines: RBM, BinaryRBM, Binary

@testset "Binary-Binary RBM partition function (brute force)" begin
    rbm = BinaryRBM(3, 2)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    randn!(rbm.w)

    Z = 0.0
    for v1 in (0,1), v2 in (0,1), v3 in (0,1)
        for h1 in (0,1), h2 in (0,1)
            v = [v1;v2;v3;;]
            h = [h1;h2;;]
            Z += exp(-only(RBMs.energy(rbm, v, h)))
        end
    end
    @test log(Z) ≈ RBMs.log_partition(rbm)

    rbm.w .= 0
    gs = Zygote.gradient(rbm) do rbm
        RBMs.log_partition(rbm)
    end
    @test only(gs).visible.par ≈ Flux.sigmoid.(rbm.visible.par)
    @test only(gs).hidden.par  ≈ Flux.sigmoid.(rbm.hidden.par)
end

@testset "Spin-Binary RBM partition function (brute force)" begin
    rbm = RBMs.RBM(RBMs.Spin((3,)), RBMs.Binary((2,)), randn(3,2))
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)

    Z = 0.0
    for v1 in (-1,1), v2 in (-1,1), v3 in (-1,1)
        for h1 in (0,1), h2 in (0,1)
            v = [v1;v2;v3;;]
            h = [h1;h2;;]
            Z += exp(-only(RBMs.energy(rbm, v, h)))
        end
    end
    @test log(Z) ≈ RBMs.log_partition(rbm)

    rbm.w .= 0
    gs = Zygote.gradient(rbm) do rbm
        RBMs.log_partition(rbm)
    end
    @test only(gs).visible.par ≈ tanh.(rbm.visible.par)
    @test only(gs).hidden.par ≈ Flux.sigmoid.(rbm.hidden.par)
end
