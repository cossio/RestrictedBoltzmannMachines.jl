using Random, Test, Zygote, SpecialFunctions, StatsFuns, FiniteDifferences,
    RestrictedBoltzmannMachines
using Zygote: @adjoint
using RestrictedBoltzmannMachines: randn_like

include("../test_utils.jl")

@testset "logerfcx adjoint" begin
    @test logerfcx'(+1) ≈ -0.63896751423479126047150115207
    @test logerfcx'(-1) ≈ -2.22527124262865745519300695143
    @test logerfcx'(+4) ≈ -0.23637689270017607216995925659
    @test logerfcx'(-4) ≈ -8.00000006349117384876269215070
    @test gradient(x -> sum(logerfcx.(x)), ones(2,2))[1] ≈ fill(logerfcx'(1),2,2)
    gradtest(x -> logerfcx.(x), [1.0 -2.0; 0.5 0.2])
end

@testset "log1pexp adjoint" begin
    @test log1pexp'(+1) ≈ 0.73105857863000487925115924182
    @test log1pexp'(-1) ≈ 0.26894142136999512074884075817
    @test log1pexp'(+4) ≈ 0.98201379003790844197320686205
    @test log1pexp'(-4) ≈ 0.01798620996209155802679313794
    @test gradient(x -> sum(log1pexp.(x)), ones(2,2))[1] ≈ fill(log1pexp'(1),2,2)
    gradtest(x -> log1pexp.(x), [1.0 -2.0; 0.5 0.2])
end

# @testset "logaddexp adjoint" begin
#     @test all(gradient(logaddexp,1,1) .≈ (0.5, 0.5))
#     @test all(gradient(logaddexp,-1,-1) .≈ (0.5, 0.5))
#     @test all(gradient(logaddexp,1,-1) .≈ (0.88079707797788244406, 0.11920292202211755594))
#     @test all(gradient(logaddexp,-1,1) .≈ (0.11920292202211755594, 0.88079707797788244406))
#     @test all(gradient((x,y) -> sum(logaddexp.(x,y)), ones(2,2),ones(2,2)) .≈ (fill(0.5,2,2), fill(0.5,2,2)))
#     gradtest((x,y) -> logaddexp.(x,y), [1.0 -2.0; 0.5 0.2], [3.0 -0.1])
# end

@testset "logaddexp" begin
    gradtest(logaddexp, randn(), randn())
    gradtest((x,y) -> logaddexp.(x,y), randn(3,3), randn(3,3))
end

@testset "sumdrop adjoint" begin
    gradtest(x -> RBMs.sumdrop(x; dims=1), randn(2,3,4))
    gradtest(x -> RBMs.sumdrop(x; dims=2), randn(2,3,4))
    gradtest(x -> RBMs.sumdrop(x; dims=3), randn(2,3,4))
    gradtest(x -> RBMs.sumdrop(x; dims=(2,3)), randn(2,3,4,2))
    gradtest(x -> RBMs.sumdrop(x; dims=(1,2,3)), randn(2,3,4))
end

@testset "sumdropfirst adjoint" begin
    gradtest(x -> RBMs.sumdropfirst(x, Val(1)), randn(1,3,4))
    gradtest(x -> RBMs.sumdropfirst(x, Val(1)), randn(1,1,4))
    gradtest(x -> RBMs.sumdropfirst(x, Val(2)), randn(1,1,4))
    gradtest(x -> RBMs.sumdropfirst(x, Val(2)), randn(1,1,1))
    gradtest(x -> RBMs.sumdropfirst(x, Val(1)), randn(2,3,4))
    gradtest(x -> RBMs.sumdropfirst(x, Val(2)), randn(2,3,4))
    gradtest(x -> RBMs.sumdropfirst(x, Val(3)), randn(2,3,4))
    gradtest(x -> RBMs.sumdropfirst(x, Val(3)), randn(2,3,4,2))
end

@testset "sqrt adjoint" begin
    @test sqrt'(1) ≈ 1/2
    @test sqrt'(4) ≈ 1/4
    gradtest(x -> sqrt.(x), [1.0 2.0; 0.5 0.2])
    gradtest(x -> sqrt.(x), rand(3,3))
end
