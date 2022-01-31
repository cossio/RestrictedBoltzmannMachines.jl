using Test, Random, LinearAlgebra, Statistics, DelimitedFiles
import Zygote, Flux, Distributions, SpecialFunctions, LogExpFunctions, QuadGK, NPZ
import RestrictedBoltzmannMachines as RBMs

@testset "regularization" begin
    rbm = RBMs.RBM(RBMs.Binary(3,5), RBMs.Gaussian(3,2), randn(3,5,3,2))
    randn!(rbm.w)
    randn!(rbm.visible.θ)
    @test RBMs.L1L2(rbm) ≈ mean(mean(abs.(rbm.w[:,:,μ,ν]))^2 for μ=1:3, ν=1:2)
    @test RBMs.pgm_reg(rbm; λv=0, λw=2) ≈ RBMs.L1L2(rbm)
    @test RBMs.pgm_reg(rbm; λv=2, λw=0) ≈ mean(rbm.visible.θ.^2)
    @inferred RBMs.L1L2(rbm)
    @inferred RBMs.pgm_reg(rbm; λv=1, λw=1)
end
