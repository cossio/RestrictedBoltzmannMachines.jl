using Test: @test, @testset
using Statistics: mean, cor
using Random: bitrand
using LogExpFunctions: softmax
using RestrictedBoltzmannMachines: RBM, Spin, Binary, Potts, Gaussian
using RestrictedBoltzmannMachines: transfer_mean, extensive_sample, zerosum
using RestrictedBoltzmannMachines: sample_v_from_h, sample_v_from_v, initialize!, pcd!, free_energy, wmean
import Flux

function train_nepochs(;
    nsamples::Int, # number observations in the data
    nupdates::Int, # desired number of parameter updates
    batchsize::Int # size of each mini-batch
)
    return ceil(Int, nupdates * batchsize / nsamples)
end

@testset "extensive_sample" begin
    @test extensive_sample(Binary(1)) == [0 1]
    @test extensive_sample(Binary(2)) == reduce(hcat, [σ1,σ2] for σ1 in 0:1, σ2 in 0:1)
    @test extensive_sample(Spin(1)) == Int8[-1 1]
    @test extensive_sample(Spin(2)) == reduce(hcat, [s1,s2] for s1 in (-1,1), s2 in (-1,1))
    @test extensive_sample(Potts(2,1)) == reshape([1,0,0,1], 2,1,2)
    @test all(sum(extensive_sample(Potts(3,4)); dims=1) .== 1)
end

@testset "pcd -- teacher/student, Binary, with weights, exact" begin
    N = 5
    batchsize = 2^N
    nupdates = 10000
    teacher = RBM(Binary(N), Binary(1), zeros(N,1))
    teacher.w[:,1] .= range(-2, 2, length=N)
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    student = RBM(Binary(N), Binary(1), zeros(N,1))
    initialize!(student, data; wts)
    student.w .= cos.(range(-10, 10, length=N))
    @test transfer_mean(student.visible) ≈ wmean(data; wts, dims=2)
    pcd!(student, data; wts, epochs, batchsize, mode=:exact, optim=Flux.AdaBelief())
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9999
end

@testset "pcd -- teacher/student, Spin, with weights, exact" begin
    N = 5
    batchsize = 2^N
    nupdates = 10000
    teacher = RBM(Spin(N), Spin(1), zeros(N,1))
    teacher.w[:,1] .= range(-1, 1, length=N)
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    student = RBM(Spin(N), Spin(1), zeros(N,1))
    initialize!(student, data; wts)
    student.w .= cos.(1:N)
    @test transfer_mean(student.visible) ≈ wmean(data; wts, dims=2)
    pcd!(student, data; wts, epochs, batchsize, mode=:exact, shuffle=false, optim=Flux.RADAM())
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9999
end

@testset "pcd -- teacher/student, Gaussian, with weights, exact" begin
    N = 5
    batchsize = 2^N
    nupdates = 10000
    teacher = RBM(Binary(N), Gaussian(1), zeros(N,1))
    teacher.w[:,1] .= range(-1, 1, length=N)
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    student = RBM(Binary(N), Gaussian(1), zeros(N,1))
    initialize!(student, data; wts)
    student.w .= cos.(1:N)
    @test transfer_mean(student.visible) ≈ wmean(data; wts, dims=2)
    pcd!(student, data; wts, epochs, batchsize, shuffle=false, mode=:exact, optim=Flux.ADAM())
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.99
end

@testset "pcd -- teacher/student, Potts, with weights, exact" begin
    q = 2
    N = 5
    batchsize = q^N
    nupdates = 2000
    teacher = RBM(Potts(q, N), Spin(1), zeros(q,N,1))
    teacher.w[1,:,1] .= range(-1, 1, length=N)
    teacher.w[2,:,1] .= -teacher.w[1,:,1]
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    student = RBM(Potts(q, N), Spin(1), zeros(q, N, 1))
    initialize!(student, data; wts)
    student.w[1,:,1] .= -student.w[2,:,1] .= cos.(1:N)
    @test transfer_mean(student.visible) ≈ wmean(data; wts, dims=3)
    pcd!(student, data; wts, epochs, batchsize, mode=:exact, optim=Flux.AdaBelief())
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9999
end

@testset "pcd -- teacher/student, Spin" begin
    N = 12
    batchsize = 8
    nupdates = 10000
    teacher = RBM(Spin(N), Spin(1), randn(N,1) ./ √N)
    # since h = ±1 are equally likely, the following gives an unbiased sample
    data = sample_v_from_h(teacher, repeat(Int8[1 -1], 1, 10000))
    student = RBM(Spin(N), Spin(1), zeros(N,1))
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    initialize!(student, data)
    @test transfer_mean(student.visible) ≈ mean(data; dims=2)
    pcd!(student, data; epochs, batchsize)
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9
end

@testset "pcd -- teacher/student, Spin, with weights" begin
    N = 12
    batchsize = 8
    nupdates = 10000
    teacher = RBM(Spin(N), Spin(1), zeros(N,1))
    teacher.w .= range(-1, 1, length=N)
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    student = RBM(Spin(N), Spin(1), zeros(N,1))
    initialize!(student, data; wts)
    @test transfer_mean(student.visible) ≈ wmean(data; wts, dims=2)
    pcd!(student, data; wts, epochs, batchsize)
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.95
end

@testset "pcd -- teacher/student, Potts" begin
    q = 2
    N = 12
    batchsize = 8
    nupdates = 10000
    teacher = RBM(Potts(q, N), Spin(1), zerosum(randn(q, N, 1)))
    # since h = ±1 are equally likely, the following gives an unbiased sample
    data = sample_v_from_h(teacher, repeat(Int8[1 -1], 1, 10000))
    student = RBM(Potts(q, N), Spin(1), zeros(q, N, 1))
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    initialize!(student, data)
    @test transfer_mean(student.visible) ≈ mean(data; dims=3)
    pcd!(student, data; epochs, batchsize)
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9
end

@testset "pcd -- teacher/student, Potts, with weights" begin
    q = 2
    N = 12
    batchsize = 8
    nupdates = 10000
    teacher = RBM(Potts(q,N), Spin(1), zeros(q,N,1))
    teacher.w[1,:,1] .= range(-1, 1, length=N)
    teacher.w[2,:,1] .= -teacher.w[1,:,1]
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    student = RBM(Potts(q,N), Spin(1), zeros(q,N,1))
    initialize!(student, data; wts)
    @test transfer_mean(student.visible) ≈ wmean(data; wts, dims=3)
    pcd!(student, data; wts, epochs, batchsize)
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.95
end

@testset "pcd -- teacher/student, Gaussian hidden" begin
    q = 2
    N = 12
    batchsize = 8
    nupdates = 10000
    teacher = RBM(Potts(q, N), Gaussian(1), zerosum(randn(q, N, 1) / 2))
    data = sample_v_from_v(teacher, falses(q, N, 10000); steps=1000)
    student = RBM(Potts(q, N), Gaussian(1), zeros(q, N, 1))
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    initialize!(student, data)
    @test transfer_mean(student.visible) ≈ mean(data; dims=3)
    pcd!(student, data; epochs, batchsize)
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9
end

@testset "pcd -- teacher/student, Potts visible, Gaussian hidden, with weights" begin
    q = 2
    N = 12
    batchsize = 16
    nupdates = 10000
    teacher = RBM(Potts(q, N), Gaussian(1), zeros(q, N, 1))
    teacher.w[1,:,1] .= range(-0.4, 0.5, length=N)
    teacher.w[2,:,1] .= -teacher.w[1,:,1]
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    student = RBM(Potts(q, N), Gaussian(1), zeros(q, N, 1))
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    initialize!(student, data)
    @test transfer_mean(student.visible) ≈ mean(data; dims=3)
    pcd!(student, data; wts, epochs, batchsize)
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9
end
