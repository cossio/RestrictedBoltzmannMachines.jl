import Random
import Zygote
import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset, @test_broken, @inferred
using Statistics: mean, var
using SpecialFunctions: erfcx
using Distributions: truncated, Normal
using RestrictedBoltzmannMachines: tnmean, tnvar, sqrt1half, randnt, randnt_half

for a = -10:10
    d = truncated(Normal(); lower=a)
    @test tnmean(a) ≈ mean(d)
    @test tnvar(a)  ≈ var(d)
    @test a ≤ tnmean(a) < Inf
    @test 0 ≤ tnvar(a) ≤ 1
end
@test 1e20 ≤ tnmean(1e20) < Inf
@test_broken 0 ≤ tnvar(1e80) ≤ 1

@testset "sqrt1half" begin
    @test (@inferred sqrt1half(5)) ≈ 5.1925824035672520156
    @test (@inferred sqrt1half(0)) == 1
    @test sqrt1half(-1) == sqrt1half(1) ≈ 1.6180339887498948482
    @test isnan(@inferred sqrt1half(NaN))
    @test sqrt1half(Inf) == sqrt1half(-Inf) == Inf
    @test sqrt1half(1e300) ≈ 1e300
end

@testset "randnt" begin
    @test (@inferred randnt(0)) > 0
    @test (@inferred randnt(1e300)) == 1e300
    @test (@inferred randnt(Inf)) == Inf
    @test isnan(@inferred randnt(NaN))
    @test (@inferred randnt(floatmax(Float64))) == floatmax(Float64)
end

Random.seed!(18)

@inferred randnt_half(1.0, 2.0)
@inferred randnt_half(Float32(1.0), Float32(2.0))
@test randnt_half(Float32(1.0), Float32(2.0)) isa Float32

# compare exact 1st and 2nd moments to Monte Carlo estimates
m1(μ,σ) = μ + σ * √(2/π) / erfcx(-μ/σ/√2)
m2(μ,σ) = μ^2 + σ^2 + μ * σ * √(2/π) / erfcx(-μ/σ/√2)

for μ = -1:1, σ = 1:2
    samples = [randnt_half(μ,σ) for _ = 1:10^6]
    @test mean(samples.^1) ≈ m1(μ,σ) atol=1e-2
    @test mean(samples.^2) ≈ m2(μ,σ) atol=1e-2
end

# broadcasted versions
μ = 3randn(2,2); σ = 3rand(2,2)
dμ, dσ = Zygote.gradient(μ,σ) do μ,σ
    mean(m1.(μ,σ))
end
