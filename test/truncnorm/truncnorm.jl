include("../tests_init.jl")

for a = -10:10
    d = truncated(Normal(), a, Inf)
    @test tnmean(a) ≈ mean(d)
    @test tnstd(a)  ≈ std(d)
    @test tnvar(a)  ≈ var(d)
    @test a ≤ tnmean(a) < Inf
    @test 0 ≤ tnvar(a) ≤ 1
end
@test 1e20 ≤ tnmean(1e20) < Inf
@test_broken 0 ≤ tnvar(1e80) ≤ 1

@testset "sqrt1half" begin
    @test sqrt1half(5) ≈ 5.1925824035672520156
    @test sqrt1half(0) == 1
    @test sqrt1half(-1) == sqrt1half(1) ≈ 1.6180339887498948482
    @test isnan(sqrt1half(NaN))
    @test sqrt1half(Inf) == sqrt1half(-Inf) == Inf
    @test sqrt1half(1e300) ≈ 1e300
end

@testset "randnt" begin
    @test randnt(0) > 0
    @test randnt(1e300) == 1e300
    @test randnt(Inf) == Inf
    @test isnan(randnt(NaN))
    @test randnt(floatmax(Float64)) == floatmax(Float64)
end
