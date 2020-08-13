using Random, Test, Zygote, SpecialFunctions, StatsFuns,
    RestrictedBoltzmannMachines
using Zygote: @adjoint

# https://fluxml.ai/Zygote.jl/latest/adjoints/#Gradient-Reflection-1
isderiving() = false
@adjoint isderiving() = true, _ -> nothing


#= Source:
https://github.com/FluxML/Zygote.jl/blob/ac4f1a0727d860b31197a336a02d04b33cb21219/test/gradcheck.jl
=#

function ngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
        δ = sqrt(eps())
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
    end
    return grads
end

function gradcheck(f, xs...)
    all(isapprox.(ngradient(f, xs...),
                  gradient(f, xs...),
                  rtol = 1e-5, atol = 1e-5))
end

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

@testset "logerfcx adjoint" begin
    @test logerfcx'(+1) ≈ -0.63896751423479126047150115207
    @test logerfcx'(-1) ≈ -2.22527124262865745519300695143
    @test logerfcx'(+4) ≈ -0.23637689270017607216995925659
    @test logerfcx'(-4) ≈ -8.00000006349117384876269215070
    @test gradient(x -> sum(logerfcx.(x)), ones(2,2))[1] ≈ fill(logerfcx'(1),2,2)
    @test gradtest(x -> logerfcx.(x), [1.0 -2.0; 0.5 0.2])
end

@testset "log1pexp adjoint" begin
    @test log1pexp'(+1) ≈ 0.73105857863000487925115924182
    @test log1pexp'(-1) ≈ 0.26894142136999512074884075817
    @test log1pexp'(+4) ≈ 0.98201379003790844197320686205
    @test log1pexp'(-4) ≈ 0.01798620996209155802679313794
    @test gradient(x -> sum(log1pexp.(x)), ones(2,2))[1] ≈ fill(log1pexp'(1),2,2)
    @test gradtest(x -> log1pexp.(x), [1.0 -2.0; 0.5 0.2])
end

# @testset "logaddexp adjoint" begin
#     @test all(gradient(logaddexp,1,1) .≈ (0.5, 0.5))
#     @test all(gradient(logaddexp,-1,-1) .≈ (0.5, 0.5))
#     @test all(gradient(logaddexp,1,-1) .≈ (0.88079707797788244406, 0.11920292202211755594))
#     @test all(gradient(logaddexp,-1,1) .≈ (0.11920292202211755594, 0.88079707797788244406))
#     @test all(gradient((x,y) -> sum(logaddexp.(x,y)), ones(2,2),ones(2,2)) .≈ (fill(0.5,2,2), fill(0.5,2,2)))
#     @test gradtest((x,y) -> logaddexp.(x,y), [1.0 -2.0; 0.5 0.2], [3.0 -0.1])
# end

@testset "logaddexp" begin
  @test gradcheck(x -> logaddexp(x[1], x[2]), [1.0, 2.0])
  @test gradcheck(x -> logaddexp(x[1], x[2]), [1.0, -1.0])
  @test gradcheck(x -> logaddexp(x[1], x[2]), [-2.0, -3.0])
  @test gradcheck(x -> logaddexp(x[1], x[2]), [5.0, 5.0])
  @test gradtest((x,y) -> logaddexp.(x,y), (3,3), (3,3))
end

@testset "sumdrop adjoint" begin
    @test gradtest(x -> RBMs.sumdrop(x; dims=1), (2,3,4))
    @test gradtest(x -> RBMs.sumdrop(x; dims=2), (2,3,4))
    @test gradtest(x -> RBMs.sumdrop(x; dims=3), (2,3,4))
    @test gradtest(x -> RBMs.sumdrop(x; dims=(2,3)), (2,3,4,2))
    @test gradtest(x -> RBMs.sumdrop(x; dims=(1,2,3)), (2,3,4))
end

@testset "sumdropfirst adjoint" begin
    @test gradtest(x -> RBMs.sumdropfirst(x, Val(1)), (1,3,4))
    @test gradtest(x -> RBMs.sumdropfirst(x, Val(1)), (1,1,4))
    @test gradtest(x -> RBMs.sumdropfirst(x, Val(2)), (1,1,4))
    @test gradtest(x -> RBMs.sumdropfirst(x, Val(2)), (1,1,1))
    @test gradtest(x -> RBMs.sumdropfirst(x, Val(1)), (2,3,4))
    @test gradtest(x -> RBMs.sumdropfirst(x, Val(2)), (2,3,4))
    @test gradtest(x -> RBMs.sumdropfirst(x, Val(3)), (2,3,4))
    @test gradtest(x -> RBMs.sumdropfirst(x, Val(3)), (2,3,4,2))
end

@testset "sqrt adjoint" begin
    @test sqrt'(+1) ≈ 1/2
    @test sqrt'(+4) ≈ 1/4
    @test gradtest(x -> sqrt.(x), [1.0 2.0; 0.5 0.2])
    @test gradcheck(x -> 2.5 * sqrt(x[1]), [1.0])
    @test gradcheck(x -> 2.5 * sqrt(x[1]), [2.45])
    @test gradtest(x -> sqrt.(x), (3,3))
end
