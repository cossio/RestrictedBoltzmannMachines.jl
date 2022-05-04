using Test: @test, @testset
using Statistics: mean, cor
using Random: bitrand
using LogExpFunctions: softmax
using RestrictedBoltzmannMachines: RBM, Spin, sample_v_from_h, initialize!, pcd!, free_energy

function train_nepochs(;
    nsamples::Int, # number observations in the data
    nupdates::Int, # desired number of parameter updates
    batchsize::Int # size of each mini-batch
)
    return ceil(Int, nupdates * batchsize / nsamples)
end

N = 10
batchsize = 16
nupdates = 50000

@testset "pcd -- teacher/student" begin
    teacher = RBM(Spin(N), Spin(1), randn(N,1))
    data = sample_v_from_h(teacher, repeat(Int8[1 -1], 1, 10000))
    student = RBM(Spin(N), Spin(1), zeros(N,1))
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    initialize!(student, data)
    pcd!(student, data; epochs, batchsize, center=false)
    @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9
end

@testset "pcd -- teacher/student, with weights" begin
    teacher = RBM(Spin(N), Spin(1), randn(N,1))
    data = Int8.(2reduce(hcat, digits.(Bool, 0:(2^N - 1), base=2, pad=N)) .- 1)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    student = RBM(Spin(N), Spin(1), zeros(N,1))
    initialize!(student, data; wts)
    pcd!(student, data; wts, epochs, batchsize, center=false)
    @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.95
end
