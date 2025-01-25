using Random: bitrand
using RestrictedBoltzmannMachines: ∂free_energy
using RestrictedBoltzmannMachines: BinaryRBM
using RestrictedBoltzmannMachines: center
using RestrictedBoltzmannMachines: CenteredBinaryRBM
using RestrictedBoltzmannMachines: energy
using RestrictedBoltzmannMachines: free_energy
using RestrictedBoltzmannMachines: initialize!
using RestrictedBoltzmannMachines: inputs_h_from_v
using RestrictedBoltzmannMachines: inputs_v_from_h
using RestrictedBoltzmannMachines: interaction_energy
using RestrictedBoltzmannMachines: mean_h_from_v
using RestrictedBoltzmannMachines: mean_v_from_h
using RestrictedBoltzmannMachines: mirror
using RestrictedBoltzmannMachines: pcd!
using RestrictedBoltzmannMachines: sample_v_from_v
using RestrictedBoltzmannMachines: uncenter
using Optimisers: Adam
using Statistics: mean
using LinearAlgebra: norm
using Test: @inferred
using Test: @test
using Test: @testset
using Zygote: gradient

@testset "CenteredBinaryRBM" begin
    rbm = @inferred CenteredBinaryRBM(randn(5), randn(3), randn(5, 3), randn(5), randn(3))
    v = bitrand(size(rbm.visible))
    h = bitrand(size(rbm.hidden))
    E = -rbm.visible.θ' * v - rbm.hidden.θ' * h - (v - rbm.offset_v)' * rbm.w * (h - rbm.offset_h)
    @test @inferred(energy(rbm, v, h)) ≈ E
    @test @inferred(inputs_v_from_h(rbm, h)) ≈ rbm.w  * (h - rbm.offset_h)
    @test @inferred(inputs_h_from_v(rbm, v)) ≈ rbm.w' * (v - rbm.offset_v)
end

@testset "center / uncenter" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    offset_v = randn(3)
    offset_h = randn(2)
    centered_rbm = @inferred center(rbm, offset_v, offset_h)
    @test centered_rbm.offset_v ≈ offset_v
    @test centered_rbm.offset_h ≈ offset_h
    @test @inferred(uncenter(centered_rbm)).visible.θ ≈ rbm.visible.θ
    @test @inferred(uncenter(centered_rbm)).hidden.θ ≈ rbm.hidden.θ
    @test @inferred(uncenter(centered_rbm)).w ≈ rbm.w

    v = bitrand(3,2)
    h = bitrand(2,2)
    @test mean_h_from_v(rbm, v) ≈ mean_h_from_v(uncenter(rbm), v)
    @test mean_v_from_h(rbm, h) ≈ mean_v_from_h(uncenter(rbm), h)
end

@testset "rbm energy invariance" begin
    centered_rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    rbm = uncenter(centered_rbm)
    ΔE = interaction_energy(rbm, centered_rbm.offset_v, centered_rbm.offset_h)::Number
    v = bitrand(size(rbm.visible)..., 100)
    h = bitrand(size(rbm.hidden)..., 100)
    @test @inferred(energy(centered_rbm, v, h)) ≈ energy(rbm, v, h) .+ ΔE
    @test @inferred(free_energy(centered_rbm, v)) ≈ free_energy(rbm, v) .+ ΔE
end

@testset "free energy" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = bitrand(size(rbm.visible)...)
    F = -log(sum(exp(-energy(rbm, v, h)) for h in [[0,0], [0,1], [1,0], [1,1]]))
    @test @inferred(free_energy(rbm, v)) ≈ F
end

@testset "∂free energy" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = bitrand(size(rbm.visible)...)
    gs = gradient(rbm) do rbm
        mean(free_energy(rbm, v))
    end
    ∂ = @inferred ∂free_energy(rbm, v)
    @test ∂.visible ≈ only(gs).visible.par
    @test ∂.hidden ≈ only(gs).hidden.par
    @test ∂.w ≈ only(gs).w
end

@testset "mirror" begin
    rbm = CenteredBinaryRBM(randn(5,2), randn(7,4,3), randn(5,2,7,4,3), randn(5,2), randn(7,4,3))
    rbm_mirror = @inferred mirror(rbm)
    @test rbm_mirror.visible == rbm.hidden
    @test rbm_mirror.hidden == rbm.visible
    @test rbm_mirror.offset_v == rbm.offset_h
    @test rbm_mirror.offset_h == rbm.offset_v
    v = rand(Bool, size(rbm.visible)..., 13)
    h = rand(Bool, size(rbm.hidden)..., 13)
    @test energy(rbm_mirror, h, v) ≈ energy(rbm, v, h)
end

@testset "centered pcd" begin
    rbm = center(BinaryRBM((28,28), 100))
    train_x = bitrand(28,28,1024)

    initialize!(rbm, train_x) # fit independent site statistics and center
    @test rbm.offset_v ≈ dropdims(mean(train_x; dims=3); dims=3)
    train_h = mean_h_from_v(rbm, train_x)
    @test rbm.offset_h ≈ vec(mean(train_h; dims=2))
    @test norm(mean(inputs_h_from_v(rbm, train_x); dims=2)) < 1e-6

    pcd!(rbm, train_x; batchsize=1024, iters=100)
    @test rbm.offset_v ≈ dropdims(mean(train_x; dims=3); dims=3)
    train_h = mean_h_from_v(rbm, train_x)
    @test rbm.offset_h ≈ vec(mean(train_h; dims=2)) rtol=0.1
    # not exact because offset is updated after having updated the parameters!
end

@testset "centered pcd training" begin
    data = falses(2, 1000)
    data[1, 1:2:end] .= true
    data[2, 1:2:end] .= true

    rbm = center(BinaryRBM(2, 5))
    initialize!(rbm, data)
    pcd!(rbm, data; iters = 10000, batchsize = 64, steps = 10, optim = Adam(5e-4))

    v_sample = sample_v_from_v(rbm, bitrand(2, 10000); steps=50)

    @test 0.4 < mean(v_sample[1,:]) < 0.6
    @test 0.4 < mean(v_sample[2,:]) < 0.6
    @test 0.4 < mean(v_sample[1,:] .* v_sample[2,:]) < 0.6
end
