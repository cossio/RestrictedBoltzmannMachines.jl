using Test: @test, @testset, @test_broken, @inferred
import Statistics
import Random
import Zygote
import Distributions
import SpecialFunctions
import RestrictedBoltzmannMachines as RBMs

for a = -10:10
    d = Distributions.truncated(Distributions.Normal(); lower=a)
    @test RBMs.tnmean(a) ≈ Statistics.mean(d)
    @test RBMs.tnstd(a)  ≈ Statistics.std(d)
    @test RBMs.tnvar(a)  ≈ Statistics.var(d)
    @test a ≤ RBMs.tnmean(a) < Inf
    @test 0 ≤ RBMs.tnvar(a) ≤ 1
end
@test 1e20 ≤ RBMs.tnmean(1e20) < Inf
@test_broken 0 ≤ RBMs.tnvar(1e80) ≤ 1

@testset "sqrt1half" begin
    @test (@inferred RBMs.sqrt1half(5)) ≈ 5.1925824035672520156
    @test (@inferred RBMs.sqrt1half(0)) == 1
    @test RBMs.sqrt1half(-1) == RBMs.sqrt1half(1) ≈ 1.6180339887498948482
    @test isnan(@inferred RBMs.sqrt1half(NaN))
    @test RBMs.sqrt1half(Inf) == RBMs.sqrt1half(-Inf) == Inf
    @test RBMs.sqrt1half(1e300) ≈ 1e300
end

@testset "randnt" begin
    @test (@inferred RBMs.randnt(0)) > 0
    @test (@inferred RBMs.randnt(1e300)) == 1e300
    @test (@inferred RBMs.randnt(Inf)) == Inf
    @test isnan(@inferred RBMs.randnt(NaN))
    @test (@inferred RBMs.randnt(floatmax(Float64))) == floatmax(Float64)
end

Random.seed!(18)

@inferred RBMs.randnt_half(1.0, 2.0)
@inferred RBMs.randnt_half(Float32(1.0), Float32(2.0))
@test RBMs.randnt_half(Float32(1.0), Float32(2.0)) isa Float32

# compare exact 1st and 2nd moments to Monte Carlo estimates
m1(μ,σ) = μ + σ * √(2/π) / SpecialFunctions.erfcx(-μ/σ/√2)
m2(μ,σ) = μ^2 + σ^2 + μ * σ * √(2/π) / SpecialFunctions.erfcx(-μ/σ/√2)

for μ = -1:1, σ = 1:2
    samples = [RBMs.randnt_half(μ,σ) for _ = 1:10^6]
    @test Statistics.mean(samples.^1) ≈ m1(μ,σ) atol=1e-2
    @test Statistics.mean(samples.^2) ≈ m2(μ,σ) atol=1e-2
end

# broadcasted versions
μ = 3randn(2,2); σ = 3rand(2,2)
dμ, dσ = Zygote.gradient(μ,σ) do μ,σ
    Statistics.mean(m1.(μ,σ))
end
