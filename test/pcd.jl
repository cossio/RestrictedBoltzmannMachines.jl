using Test: @test, @testset
using Statistics: mean, cor
using Random: bitrand
using RestrictedBoltzmannMachines: RBM, Spin, sample_v_from_v, initialize!, pcd!, free_energy

@testset "pcd -- teacher - student" begin
    teacher = RBM(Spin(10), Spin(1), randn(10,1))
    data = sample_v_from_v(teacher, zeros(Int8, 10, 10000); steps=100)
    student = RBM(Spin(10), Spin(1), zeros(10,1))
    initialize!(student, data)
    pcd!(student, data; epochs=10, batchsize=128)
    cor(free_energy(teacher, data), free_energy(student, data))
    @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9
end
