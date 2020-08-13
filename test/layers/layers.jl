using Test, Random, Statistics, NNlib, StatsFuns
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: meandrop

Random.seed!(785)

@testset "Testing $Layer" for Layer in (Binary, Spin, Gaussian, ReLU, dReLU)
    layer = Layer(3,4,5)
    @test size(layer) == (3,4,5)
    I = randn(3,4,5, 10,2)
    x = random(layer, I, 2.0)
    @test size(x) == size(I)

    # single batch case
    @test energy(layer, random(layer)) isa Number
    @test cgf(layer, randn(size(layer))) isa Number

    @test sitedims(layer) == (1,2,3)
    @test batchdims(layer, x) == (4,5)
    @inferred sitedims(layer)
    @inferred batchdims(layer, x)

    @test sitesize(layer) == size(layer) == (3,4,5)
    @test batchsize(layer, x) == (10,2)
    @inferred sitesize(layer)
    @inferred batchsize(layer, x)

    @test siteindices(layer) == CartesianIndices(zeros(3,4,5))
    @test batchindices(layer, x) == CartesianIndices(zeros(10,2))
    @inferred siteindices(layer)
    @inferred batchindices(layer, x)

    @test x[siteindices(layer), batchindices(layer, x)] == x
    @inferred x[siteindices(layer), batchindices(layer, x)]
    @test x[siteindices(layer),4,1] == x[:,:,:,4,1]
    @test x[2,1,3,batchindices(layer,x)] == x[2,1,3,:,:]
    @inferred x[siteindices(layer),4,1]
    @inferred x[2,1,3,batchindices(layer,x)]
    @test x[first(siteindices(layer)), first(batchindices(layer,x))] == first(x)
    @inferred x[first(siteindices(layer)), first(batchindices(layer,x))]

    @test size(cgf(layer, I)) == (10, 2)
    @test size(energy(layer, x)) == (10, 2)
    @test size(transfer_mode(layer, I)) == size(x)
    @inferred energy(layer, x)
    @inferred cgf(layer, I)
    @inferred transfer_mode(layer, I)
    @inferred random(layer, I, 2.0)

    I = randn(size(layer)...)
    β = 2.0
    samples = [random(layer, I, β) for _ = 1:10^4]
    @test mean(samples) ≈ transfer_mean(layer, I, β) atol=0.1
    @test std(samples) ≈ transfer_std(layer, I, β)  atol=0.1
end
