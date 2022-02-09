using Test: @test, @testset
import Statistics
import Random
import Zygote
import RestrictedBoltzmannMachines as RBMs

@testset "subtract_gradients" begin
    nt1 = (x = [2], y = [3])
    nt2 = (x = [1], y = [-1])
    @test RBMs.subtract_gradients(nt1, nt2) == (x = [1], y = [4])

    nt1 = (x = [2], y = [3], t = (a = [1], b = [2]))
    nt2 = (x = [1], y = [-1], t = (a = [2], b = [0]))
    @test RBMs.subtract_gradients(nt1, nt2) == (
        x = [1], y = [4], t = (a = [-1], b = [2])
    )
end

@testset "∂free_energy(rbm, v)" begin
    rbm = RBMs.BinaryRBM(randn(5,2), randn(4,3), randn(5,2,4,3))
    v = Random.bitrand(size(rbm.visible)..., 7)
    gs = Zygote.gradient(rbm) do rbm
        Statistics.mean(RBMs.free_energy(rbm, v))
    end
    ∂F = RBMs.∂free_energy(rbm, v)
    @test ∂F.visible.θ ≈ only(gs).visible.θ
    @test ∂F.hidden.θ ≈ only(gs).hidden.θ
    @test ∂F.w ≈ only(gs).w
end

@testset "∂contrastive_divergence" begin
    rbm = RBMs.BinaryRBM(randn(5,2), randn(4,3), randn(5,2,4,3))
    vd = Random.bitrand(size(RBMs.visible(rbm))..., 7)
    vm = Random.bitrand(size(RBMs.visible(rbm))..., 7)
    gs = Zygote.gradient(rbm) do rbm
        RBMs.contrastive_divergence(rbm, vd, vm)
    end
    ∂ = RBMs.∂contrastive_divergence(rbm, vd, vm)
    @test ∂.visible.θ ≈ only(gs).visible.θ
    @test ∂.hidden.θ ≈ only(gs).hidden.θ
    @test ∂.w ≈ only(gs).w
end