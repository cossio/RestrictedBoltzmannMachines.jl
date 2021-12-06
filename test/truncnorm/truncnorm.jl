include("../tests_init.jl")

for a = -10:10
    d = Distributions.truncated(Distributions.Normal(), a, Inf)
    @test RBMs.tnmean(a) ≈ mean(d)
    @test RBMs.tnstd(a)  ≈ std(d)
    @test RBMs.tnvar(a)  ≈ var(d)
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
