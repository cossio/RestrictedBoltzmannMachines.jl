import Zygote
import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset
using Random: bitrand
using Statistics: mean

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
    v = bitrand(size(rbm.visible)..., 7)
    gs = Zygote.gradient(rbm) do rbm
        mean(RBMs.free_energy(rbm, v))
    end
    ∂F = RBMs.∂free_energy(rbm, v)
    @test ∂F.visible.θ ≈ only(gs).visible.θ
    @test ∂F.hidden.θ ≈ only(gs).hidden.θ
    @test ∂F.w ≈ only(gs).w

    wts = rand(7)
    gs = Zygote.gradient(rbm) do rbm
        RBMs.wmean(RBMs.free_energy(rbm, v); wts)
    end
    ∂F = RBMs.∂free_energy(rbm, v; wts)
    @test ∂F.visible.θ ≈ only(gs).visible.θ
    @test ∂F.hidden.θ ≈ only(gs).hidden.θ
    @test ∂F.w ≈ only(gs).w
end

@testset "∂contrastive_divergence" begin
    rbm = RBMs.BinaryRBM(randn(5,2), randn(4,3), randn(5,2,4,3))
    vd = bitrand(size(RBMs.visible(rbm))..., 7)
    vm = bitrand(size(RBMs.visible(rbm))..., 7)
    gs = Zygote.gradient(rbm) do rbm
        RBMs.contrastive_divergence(rbm, vd, vm)
    end
    ∂ = RBMs.∂contrastive_divergence(rbm, vd, vm)
    @test ∂.visible.θ ≈ only(gs).visible.θ
    @test ∂.hidden.θ ≈ only(gs).hidden.θ
    @test ∂.w ≈ only(gs).w
end
