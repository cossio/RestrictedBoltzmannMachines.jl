using Test: @test, @testset
using Statistics: mean, cor
using LinearAlgebra: norm, Diagonal
using Random: bitrand
using LogExpFunctions: softmax
using RestrictedBoltzmannMachines: RBM, Spin, Binary, Potts, Gaussian, training_epochs
using RestrictedBoltzmannMachines: mean_h_from_v, var_h_from_v, batchmean, batchvar
using RestrictedBoltzmannMachines: mean_from_inputs, extensive_sample, zerosum
using RestrictedBoltzmannMachines: sample_v_from_h, sample_v_from_v, initialize!, pcd!, free_energy, wmean
import Flux

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
    epochs = training_epochs(; nsamples, batchsize, nupdates)
    student = RBM(Binary(N), Binary(1), zeros(N,1))
    initialize!(student, data; wts)
    student.w .= cos.(range(-10, 10, length=N))
    @test mean_from_inputs(student.visible) ≈ wmean(data; wts, dims=2)
    pcd!(student, data; wts, epochs, batchsize, mode=:exact, optim=Flux.AdaBelief())
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9999

    # moment matching conditions
    wts_student = softmax(-free_energy(student, data))

    v_student = batchmean(student.visible, data; wts = wts_student)
    v_teacher = batchmean(teacher.visible, data; wts)
    @info @test norm(v_student - v_teacher) < 1e-10

    h_student = batchmean(student.hidden, mean_h_from_v(student, data); wts=wts_student)
    h_teacher = batchmean(teacher.hidden, mean_h_from_v(student, data); wts)
    @info @test norm(h_student - h_teacher) < 1e-10

    vh_student = data * Diagonal(wts_student) * mean_h_from_v(student, data)' / sum(wts_student)
    vh_teacher = data * Diagonal(wts) * mean_h_from_v(student, data)' / sum(wts)
    @info @test norm(vh_student - vh_teacher) < 1e-10
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
    epochs = training_epochs(; nsamples, batchsize, nupdates)
    student = RBM(Spin(N), Spin(1), zeros(N,1))
    initialize!(student, data; wts)
    student.w .= cos.(1:N)
    @test mean_from_inputs(student.visible) ≈ wmean(data; wts, dims=2)
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
    epochs = training_epochs(; nsamples, batchsize, nupdates)
    student = RBM(Binary(N), Gaussian(1), zeros(N,1))
    initialize!(student, data; wts)
    student.w .= cos.(1:N)
    @test mean_from_inputs(student.visible) ≈ wmean(data; wts, dims=2)
    pcd!(student, data; wts, epochs, batchsize, ϵh=1e-2, shuffle=false, mode=:exact, optim=Flux.ADAM())
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.99
    wts_student = softmax(-free_energy(student, data))
    ν_int = batchmean(student.hidden, var_h_from_v(student, data); wts = wts_student)
    ν_ext = batchvar(student.hidden, mean_h_from_v(student, data); wts = wts_student)
    @test only(ν_int + ν_ext) ≈ 1 - 1e-2 # not exactly 1 because of ϵh
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
    epochs = training_epochs(; nsamples, batchsize, nupdates)
    student = RBM(Potts(q, N), Spin(1), zeros(q, N, 1))
    initialize!(student, data; wts)
    student.w[1,:,1] .= -student.w[2,:,1] .= cos.(1:N)
    @test mean_from_inputs(student.visible) ≈ wmean(data; wts, dims=3)
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
    epochs = training_epochs(; nsamples, batchsize, nupdates)
    initialize!(student, data)
    @test mean_from_inputs(student.visible) ≈ mean(data; dims=2)
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
    epochs = training_epochs(; nsamples, batchsize, nupdates)
    student = RBM(Spin(N), Spin(1), zeros(N,1))
    initialize!(student, data; wts)
    @test mean_from_inputs(student.visible) ≈ wmean(data; wts, dims=2)
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
    epochs = training_epochs(; nsamples, batchsize, nupdates)
    initialize!(student, data)
    @test mean_from_inputs(student.visible) ≈ mean(data; dims=3)
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
    epochs = training_epochs(; nsamples, batchsize, nupdates)
    student = RBM(Potts(q,N), Spin(1), zeros(q,N,1))
    initialize!(student, data; wts)
    @test mean_from_inputs(student.visible) ≈ wmean(data; wts, dims=3)
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
    epochs = training_epochs(; nsamples, batchsize, nupdates)
    initialize!(student, data)
    @test mean_from_inputs(student.visible) ≈ mean(data; dims=3)
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
    epochs = training_epochs(; nsamples, batchsize, nupdates)
    initialize!(student, data)
    @test mean_from_inputs(student.visible) ≈ mean(data; dims=3)
    pcd!(student, data; wts, epochs, batchsize)
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9
end