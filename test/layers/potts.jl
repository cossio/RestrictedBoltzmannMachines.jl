using Test, Random, LinearAlgebra, Statistics
using NNlib, StatsFuns, Zygote, FiniteDifferences
using RestrictedBoltzmannMachines, OneHot
using RestrictedBoltzmannMachines: meandrop

Random.seed!(788)

layer = Potts(3,4,2)
randn!(layer.θ)
β = 2.0
I = randn(3,4,2, 3,2)
@test layer.q == 3
x = random(layer, I, β)
@test size(I) == size(x)

# single batch case
@test energy(layer, random(layer)) isa Number
@test cgf(layer, randn(size(layer))) isa Number

@test sitedims(layer) == (2,3)
@test batchdims(layer, x) == (4,5)
@inferred sitedims(layer)
@inferred batchdims(layer, x)

@test sitesize(layer) == (4,2)
@test batchsize(layer, x) == (3,2)
@inferred sitesize(layer)
@inferred batchsize(layer, x)

@test siteindices(layer) == CartesianIndices(zeros(4,2))
@test batchindices(layer, x) == CartesianIndices(zeros(3,2))
@inferred siteindices(layer)
@inferred batchindices(layer, x)

@test x[:, siteindices(layer), batchindices(layer, x)] == x
@inferred x[:,siteindices(layer), batchindices(layer, x)]
@test x[1,siteindices(layer),2,1] == x[1,:,:,2,1]
@test x[2,1,2,batchindices(layer,x)] == x[2,1,2,:,:]
@inferred x[1,siteindices(layer),2,1]
@inferred x[2,1,2,batchindices(layer,x)]
@test x[1,siteindices(layer)[1], batchindices(layer,x)[1]] == x[1]
@inferred x[1,siteindices(layer)[1], batchindices(layer,x)[1]]

@test size(energy(layer, x)) == (3, 2)
@test size(cgf(layer, I)) == (3, 2)
@test size(transfer_mode(layer, I)) == size(I)
@test size(transfer_mode(layer)) == size(layer)

@inferred random(layer, I, β)
@inferred cgf(layer, I)
@inferred energy(layer, x)
@inferred transfer_mode(layer, I)

x = random(layer, I)
@test x isa Array{Float64}
all(iszero.(x) .| isone.(x))
for b=1:2, a=1:3, j=1:2, i=1:4
    @test count(isone, x[:, i,j, a,b]) == 1
    @test count(!isone, x[:, i,j, a,b]) == layer.q - 1
    @test count(iszero, x[:, i,j, a,b]) == layer.q - 1
end

x = transfer_mode(layer, I)
@test x isa AbstractArray{Float64}
all(iszero.(x) .| isone.(x))
for b=1:2, a=1:3, j=1:2, i=1:4
    @test count(isone, x[:, i,j, a,b]) == 1
    o = argmax((layer.θ .+ I)[:, i,j, a,b])
    @test isone(x[o, i,j, a,b])
end

Γ = StatsFuns.logsumexp(β .* (layer.θ .+ I); dims=1) ./ β
@test cgf(layer, I, β) ≈ [sum(Γ[:,:,:,a,b]) for a=1:3, b=1:2]

layer = Potts(2,1)
randn!(layer.θ)
I = randn(size(layer))
@test size(random(layer, I, β)) == size(I)
avg = zeros(size(I))
for _ = 1:10^4
    avg .+= random(layer, I, β) ./ 10^4
end
p = NNlib.softmax(β .* (layer.θ .+ I); dims=1)
@test p ≈ avg atol=0.1

layer = Potts(3,4,2)
randn!(layer.θ)
I = randn(3,4,2, 1024)
x = random(layer, I)
@test size(I) == size(x)
init!(layer, x; eps=0)
@test OneHot.softmax(layer.θ) ≈ meandrop(x; dims=(4,))
@test all(abs.(mean(layer.θ; dims=1)) .< 1e-10)

@testset "Potts energy gradients" begin
    # with batch dimensions
    θ = randn(4,5,6)
    x = rand(Bool, 4,5,6, 3,2)
    testfun(θ::AbstractArray) = sum(energy(Potts(θ), x))
    (dθ,) = gradient(testfun, θ)
    p = randn(size(θ))
    @test central_fdm(5,1)(ϵ -> testfun(θ + ϵ * p), 0) ≈ sum(dθ .* p)

    # without batch dimensions
    θ = randn(4,5,6)
    x = rand(Bool, 4,5,6)
    testfun(θ::AbstractArray) = sum(energy(Potts(θ), x))
    (dθ,) = gradient(testfun, θ)
    p = randn(size(θ))
    @test central_fdm(5,1)(ϵ -> testfun(θ + ϵ * p), 0) ≈ sum(dθ .* p)
end

layer = Potts(randn(10,5))
sample = random(layer, zeros(size(layer)..., 10000))
@test transfer_mean(layer) ≈ mean(sample; dims=3) rtol=0.1
@test transfer_mean_abs(layer) ≈ mean(abs, sample; dims=3) rtol=0.1

@testset "Potts pdf" begin
    configs = OneHot.encode.(vec(collect(RBMs.seqgen(3, 1:3))), 3)
    layer = Potts(randn(3,3))
    @test sum(transfer_pdf(layer, x) for x in configs) ≈ 1
end