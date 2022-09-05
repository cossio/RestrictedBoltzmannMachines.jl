import Zygote
import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset
using Random: bitrand
using Statistics: mean
using LinearAlgebra: norm
using RestrictedBoltzmannMachines: BinaryRBM

@testset "∂free_energy(rbm, v)" begin
    rbm = RBMs.BinaryRBM(randn(5,2), randn(4,3), randn(5,2,4,3))
    v = bitrand(size(rbm.visible)..., 7)
    gs = Zygote.gradient(rbm) do rbm
        mean(RBMs.free_energy(rbm, v))
    end
    ∂F = RBMs.∂free_energy(rbm, v)
    @test ∂F.visible ≈ only(gs).visible.par
    @test ∂F.hidden ≈ only(gs).hidden.par
    @test ∂F.w ≈ only(gs).w

    gnorms = RBMs.gradnorms(∂F)
    @test gnorms.visible ≈ norm(∂F.visible)
    @test gnorms.hidden ≈ norm(∂F.hidden)
    @test gnorms.w ≈ norm(∂F.w)

    gλ = 2.3 * ∂F
    @test gλ.visible ≈ ∂F.visible * 2.3
    @test gλ.hidden ≈ ∂F.hidden * 2.3
    @test gλ.w ≈ ∂F.w * 2.3

    wts = rand(7)
    gs = Zygote.gradient(rbm) do rbm
        RBMs.wmean(RBMs.free_energy(rbm, v); wts)
    end
    ∂F = RBMs.∂free_energy(rbm, v; wts)
    @test ∂F.visible ≈ only(gs).visible.par
    @test ∂F.hidden ≈ only(gs).hidden.par
    @test ∂F.w ≈ only(gs).w
end

@testset "∂contrastive_divergence" begin
    rbm = RBMs.BinaryRBM(randn(5,2), randn(4,3), randn(5,2,4,3))
    vd = bitrand(size(rbm.visible)..., 7)
    vm = bitrand(size(rbm.visible)..., 7)
    gs = Zygote.gradient(rbm) do rbm
        RBMs.contrastive_divergence(rbm, vd, vm)
    end
    ∂ = RBMs.∂contrastive_divergence(rbm, vd, vm)
    @test ∂.visible ≈ only(gs).visible.par
    @test ∂.hidden ≈ only(gs).hidden.par
    @test ∂.w ≈ only(gs).w
end
