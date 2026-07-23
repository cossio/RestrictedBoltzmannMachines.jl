import Random
using Test: @test, @testset
using Statistics: mean
using Random: bitrand
using Optimisers: Adam
using RestrictedBoltzmannMachines: RBM, BinaryRBM, HopfieldRBM, sample_v_from_v, initialize!, pcd!

Random.seed!(23)

@testset "pcd" begin
    data = falses(2, 1000)
    data[1, 1:2:end] .= true
    data[2, 1:2:end] .= true

    rbm = BinaryRBM(2, 5)
    initialize!(rbm, data)
    pcd!(rbm, data; iters = 10000, batchsize = 64, steps = 10, optim = Adam(5.0e-4))

    v_sample = sample_v_from_v(rbm, bitrand(2, 10000); steps = 50)

    @test 0.4 < mean(v_sample[1, :]) < 0.6
    @test 0.4 < mean(v_sample[2, :]) < 0.6
    @test 0.4 < mean(v_sample[1, :] .* v_sample[2, :]) < 0.6
end

@testset "default pcd! with zero-initialized continuous hidden units" begin
    data = Float64[-1 1 -1 1; -1 -1 1 1]
    rbm = HopfieldRBM(2, 1)

    pcd!(rbm, data)

    @test all(isfinite, rbm.visible.par)
    @test all(isfinite, rbm.hidden.par)
    @test all(isfinite, rbm.w)
end

@testset "pcd! unified callback keywords" begin
    data = bitrand(2, 32)
    rbm = BinaryRBM(2, 3)
    seen = Ref{Any}(nothing)
    state, ps = pcd!(rbm, data; iters = 2, batchsize = 8, callback = (; kwargs...) -> (seen[] = kwargs))
    @test issubset((:rbm, :optim, :state, :ps, :iter, :vd, :wd, :∂, :vm), keys(seen[]))
    @test seen[][:rbm] === rbm
    @test seen[][:ps] === ps
    @test seen[][:state] == state
end
