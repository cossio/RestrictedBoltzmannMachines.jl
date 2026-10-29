import Random
using Test: @test, @testset, @inferred
using Statistics: mean, var
using SpecialFunctions: erfcx
using Distributions: truncated, Normal
using RestrictedBoltzmannMachines: tnmean, tnvar, tnmeanvar, sqrt1half, randnt, randnt_half

mutable struct TruncatedNormalTestRNG <: Random.AbstractRNG
    normals::Vector{Float64}
    exponentials::Vector{Float64}
    uniforms::Vector{Float64}
end

Random.randn(rng::TruncatedNormalTestRNG, ::Type{Float64}) = popfirst!(rng.normals)
Random.randexp(rng::TruncatedNormalTestRNG, ::Type{Float64}) = popfirst!(rng.exponentials)
Random.rand(rng::TruncatedNormalTestRNG, ::Type{Float64}) = popfirst!(rng.uniforms)

for a in -10:10
    d = truncated(Normal(); lower = a)
    @test tnmean(a) ≈ mean(d)
    @test tnvar(a) ≈ var(d)
    @test a ≤ tnmean(a) < Inf
    @test 0 ≤ tnvar(a) ≤ 1
end
@test 1.0e20 ≤ tnmean(1.0e20) < Inf
@test tnvar(1.0e80) ≈ 1.0e-160

@testset "sqrt1half" begin
    @test (@inferred sqrt1half(5)) ≈ 5.1925824035672520156
    @test (@inferred sqrt1half(0)) == 1
    @test sqrt1half(-1) == sqrt1half(1) ≈ 1.6180339887498948482
    @test isnan(@inferred sqrt1half(NaN))
    @test sqrt1half(Inf) == sqrt1half(-Inf) == Inf
    @test sqrt1half(1.0e300) ≈ 1.0e300
end

@testset "randnt" begin
    @test (@inferred randnt(0)) > 0
    @test (@inferred randnt(1.0e300)) == 1.0e300
    @test (@inferred randnt(Inf)) == Inf
    @test isnan(@inferred randnt(NaN))
    @test (@inferred randnt(floatmax(Float64))) == floatmax(Float64)
end

@testset "randnt draw sequences" begin
    rng = TruncatedNormalTestRNG([-1.0, 0.0], Float64[], Float64[])
    @test randnt(rng, -0.5) == 0.0
    @test isempty(rng.normals)

    t = sqrt1half(1.0)
    rng = TruncatedNormalTestRNG(Float64[], [10.0, 1.0], [0.5, 0.5])
    @test randnt(rng, 1.0) == 1.0 + 1.0 / t
    @test isempty(rng.exponentials)
    @test isempty(rng.uniforms)

    rng = TruncatedNormalTestRNG(Float64[], Float64[], Float64[])
    @test randnt(rng, Inf) == Inf
    @test isnan(randnt(rng, NaN))
end

Random.seed!(18)

@inferred randnt_half(1.0, 2.0)
@inferred randnt_half(Float32(1.0), Float32(2.0))
@test randnt_half(Float32(1.0), Float32(2.0)) isa Float32

# compare exact 1st and 2nd moments to Monte Carlo estimates
m1(μ, σ) = μ + σ * √(2 / π) / erfcx(-μ / σ / √2)
m2(μ, σ) = μ^2 + σ^2 + μ * σ * √(2 / π) / erfcx(-μ / σ / √2)

for μ in -1:1, σ in 1:2
    samples = [randnt_half(μ, σ) for _ in 1:(10^6)]
    @test mean(samples .^ 1) ≈ m1(μ, σ) atol = 1.0e-2
    @test mean(samples .^ 2) ≈ m2(μ, σ) atol = 1.0e-2
end

@testset "randnt argument conversion" begin
    rng = Random.default_rng()
    @test @inferred(randnt(rng, 1)) ≥ 1       # Int argument
    @test @inferred(randnt(rng, big"1.5")) ≥ 1.5  # BigFloat argument
    @test randnt(rng, big"1.5") isa Float64
end

@testset "tnmeanvar" begin
    for a in -2:0.5:2
        μ, ν = @inferred tnmeanvar(a)
        @test μ ≈ tnmean(a)
        @test ν ≈ tnvar(a)
    end
    # on arrays, a tuple of arrays rather than an array of tuples
    a = collect(-2:0.5:2)
    μ, ν = @inferred tnmeanvar(a)
    @test μ ≈ tnmean.(a)
    @test ν ≈ tnvar.(a)
end

@testset "tnmeanvar with location and scale" begin
    for μ in (-2.0, 0.0, 3.0), σ in (0.5, 1.0, 2.0)
        d = truncated(Normal(μ, σ); lower = 0)
        m, v = @inferred tnmeanvar(μ, σ)
        @test m ≈ mean(d)
        @test v ≈ var(d)
    end
    for T in (Float32, Float64)
        m, v = @inferred tnmeanvar(T(-77), T(1))
        @test m isa T
        @test v isa T
        @test m ≈ T(0.012982635832510166) rtol = 8eps(T)
        @test v ≈ T(0.00016849206355774916) rtol = 8eps(T)
    end
end

@testset "truncated-normal tail moments" begin
    setprecision(BigFloat, 256) do
        for T in (Float32, Float64), a in T.((8, 10, 20, 77, 100, 1000, 1.0e8))
            b = BigFloat(a)
            μ_ref = sqrt(2 / big(π)) / erfcx(b / sqrt(big(2)))
            ν_ref = 1 - (μ_ref - b) * μ_ref
            μ, ν = @inferred tnmeanvar(a)
            @test μ isa T
            @test ν isa T
            @test μ ≈ T(μ_ref) rtol = 8eps(T)
            @test ν ≈ T(ν_ref) rtol = 8eps(T)
            @test 0 < ν < 1
        end
        for T in (Float32, Float64)
            a = reshape(T[-10, 0, 2, prevfloat(T(8)), 8, nextfloat(T(8)), 77, 1000], 2, 4)
            μ, ν = @inferred tnmeanvar(a)
            @test size(μ) == size(ν) == size(a)
            @test μ == first.(tnmeanvar.(a))
            @test ν == last.(tnmeanvar.(a))
            for i in eachindex(a)
                b = BigFloat(a[i])
                μ_ref = sqrt(2 / big(π)) / erfcx(b / sqrt(big(2)))
                ν_ref = 1 - (μ_ref - b) * μ_ref
                @test μ[i] ≈ T(μ_ref) rtol = 64eps(T)
                @test ν[i] ≈ T(ν_ref) rtol = 8192eps(T)
            end
        end
    end
    for bits in (256, 512)
        setprecision(BigFloat, bits) do
            a = BigFloat(8)
            μ_ref = sqrt(2 / big(π)) / erfcx(a / sqrt(big(2)))
            μ, ν = tnmeanvar(a)
            @test μ ≈ μ_ref rtol = 8eps(a)
            @test ν ≈ 1 - (μ_ref - a) * μ_ref rtol = 8192eps(a)
        end
    end
end
