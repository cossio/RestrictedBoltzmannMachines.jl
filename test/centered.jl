using Random: bitrand
using RestrictedBoltzmannMachines: ∂free_energy
using RestrictedBoltzmannMachines: ∂regularize!
using RestrictedBoltzmannMachines: batchmean
using RestrictedBoltzmannMachines: BinaryRBM
using RestrictedBoltzmannMachines: center
using RestrictedBoltzmannMachines: center!
using RestrictedBoltzmannMachines: center_from_data!
using RestrictedBoltzmannMachines: center_hidden
using RestrictedBoltzmannMachines: center_hidden!
using RestrictedBoltzmannMachines: center_hidden_from_data!
using RestrictedBoltzmannMachines: center_visible
using RestrictedBoltzmannMachines: center_visible!
using RestrictedBoltzmannMachines: center_visible_from_data!
using RestrictedBoltzmannMachines: CenteredBinaryRBM
using RestrictedBoltzmannMachines: CenteredRBM
using RestrictedBoltzmannMachines: ReLU
using RestrictedBoltzmannMachines: dReLU
using RestrictedBoltzmannMachines: Binary
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
using RestrictedBoltzmannMachines: sample_h_from_h
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

    rbm_ = deepcopy(rbm)
    offset_v = randn(size(rbm.visible)...)
    offset_h = randn(size(rbm.hidden)...)
    center!(rbm_, offset_v, offset_h)
    @test rbm_.visible.θ == center(rbm, offset_v, offset_h).visible.θ
    @test rbm_.hidden.θ == center(rbm, offset_v, offset_h).hidden.θ
    @test rbm_.w == center(rbm, offset_v, offset_h).w
    @test rbm_.offset_v == center(rbm, offset_v, offset_h).offset_v == offset_v
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

@testset "center_visible / center_hidden helpers" begin
    rbm = center(BinaryRBM(randn(3), randn(2), randn(3,2)))
    offset_v = randn(3)
    offset_h = randn(2)

    rbm_visible = @inferred center_visible(rbm, offset_v)
    rbm_hidden = @inferred center_hidden(rbm, offset_h)
    rbm_visible_ref = center(rbm, offset_v, rbm.offset_h)
    rbm_hidden_ref = center(rbm, rbm.offset_v, offset_h)
    @test rbm_visible.visible.par == rbm_visible_ref.visible.par
    @test rbm_visible.hidden.par == rbm_visible_ref.hidden.par
    @test rbm_visible.w == rbm_visible_ref.w
    @test rbm_visible.offset_v == rbm_visible_ref.offset_v
    @test rbm_visible.offset_h == rbm_visible_ref.offset_h
    @test rbm_hidden.visible.par == rbm_hidden_ref.visible.par
    @test rbm_hidden.hidden.par == rbm_hidden_ref.hidden.par
    @test rbm_hidden.w == rbm_hidden_ref.w
    @test rbm_hidden.offset_v == rbm_hidden_ref.offset_v
    @test rbm_hidden.offset_h == rbm_hidden_ref.offset_h

    rbm_visible_mut = deepcopy(rbm)
    @test center_visible!(rbm_visible_mut, offset_v) === rbm_visible_mut
    @test rbm_visible_mut.visible.par == rbm_visible.visible.par
    @test rbm_visible_mut.hidden.par == rbm_visible.hidden.par
    @test rbm_visible_mut.w == rbm_visible.w
    @test rbm_visible_mut.offset_v == rbm_visible.offset_v
    @test rbm_visible_mut.offset_h == rbm_visible.offset_h

    rbm_hidden_mut = deepcopy(rbm)
    @test center_hidden!(rbm_hidden_mut, offset_h) === rbm_hidden_mut
    @test rbm_hidden_mut.visible.par == rbm_hidden.visible.par
    @test rbm_hidden_mut.hidden.par == rbm_hidden.hidden.par
    @test rbm_hidden_mut.w == rbm_hidden.w
    @test rbm_hidden_mut.offset_v == rbm_hidden.offset_v
    @test rbm_hidden_mut.offset_h == rbm_hidden.offset_h
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
    rbm = CenteredBinaryRBM(
        randn(5,2), randn(7,4,3), randn(5,2,7,4,3), randn(5,2), randn(7,4,3)
    )
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

@testset "center_from_data! helpers" begin
    rbm = center(BinaryRBM(randn(3), randn(2), randn(3,2)))
    data = bitrand(3, 7)
    wts = rand(7)

    rbm_visible = deepcopy(rbm)
    @test center_visible_from_data!(rbm_visible, data; wts) === rbm_visible
    expected_offset_v = batchmean(rbm.visible, data; wts)
    @test rbm_visible.offset_v ≈ expected_offset_v

    expected_hidden = center_visible(rbm, expected_offset_v)
    expected_offset_h = batchmean(expected_hidden.hidden, mean_h_from_v(expected_hidden, data); wts)

    rbm_hidden = deepcopy(expected_hidden)
    @test center_hidden_from_data!(rbm_hidden, data; wts) === rbm_hidden
    @test rbm_hidden.offset_h ≈ expected_offset_h

    rbm_data = deepcopy(rbm)
    @test center_from_data!(rbm_data, data; wts) === rbm_data
    @test rbm_data.visible.par == rbm_hidden.visible.par
    @test rbm_data.hidden.par == rbm_hidden.hidden.par
    @test rbm_data.w == rbm_hidden.w
    @test rbm_data.offset_v == rbm_hidden.offset_v
    @test rbm_data.offset_h == rbm_hidden.offset_h
end

@testset "sample_h_from_h centered RBM" begin
    rbm = center(BinaryRBM(randn(3), randn(2), zeros(3,2)))
    h = bitrand(2, 10^5)
    sample = @inferred sample_h_from_h(rbm, h)
    @test size(sample) == size(h)
    @test batchmean(rbm.hidden, sample) ≈ batchmean(rbm.hidden, mean_h_from_v(rbm, bitrand(3, 10^5))) rtol=0.1
end

@testset "∂regularize! centered RBM" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = bitrand(3, 100)
    vdims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    urbm = uncenter(rbm)
    ∂u = ∂free_energy(urbm, v)

    gs = gradient(rbm) do rbm
        F = mean(free_energy(rbm, v))
        urbm = uncenter(rbm)
        L2_fields = sum(abs2, urbm.visible.θ)
        L1_weights = sum(abs, urbm.w)
        L2_weights = sum(abs2, urbm.w)
        L2L1_weights = sum(abs2, sum(abs, urbm.w; dims=vdims))
        return (
            F + l2_fields/2 * L2_fields +
            l1_weights * L1_weights +
            l2_weights/2 * L2_weights +
            l2l1_weights/(2N) * L2L1_weights
        )
    end

    ∂ = ∂free_energy(rbm, v)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
    @test only(gs).w ≈ ∂.w
end

@testset "∂regularize! centered RBM" begin
    rbm = CenteredRBM(
        ReLU(; θ=randn(3), γ=rand(3)), Binary(; θ=randn(2)), randn(3,2),
        randn(3), randn(2)
    )
    v = rand(3, 100)
    vdims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    gs = gradient(rbm) do rbm
        F = mean(free_energy(rbm, v))
        urbm = uncenter(rbm)
        L2_fields = sum(abs2, urbm.visible.θ)
        L1_weights = sum(abs, urbm.w)
        L2_weights = sum(abs2, urbm.w)
        L2L1_weights = sum(abs2, sum(abs, urbm.w; dims=vdims))
        return (
            F + l2_fields/2 * L2_fields +
            l1_weights * L1_weights +
            l2_weights/2 * L2_weights +
            l2l1_weights/(2N) * L2L1_weights
        )
    end

    ∂ = ∂free_energy(rbm, v)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
    @test only(gs).w ≈ ∂.w
end

@testset "∂regularize! centered RBM" begin
    rbm = CenteredRBM(
        dReLU(; θp=randn(3), θn=randn(3), γp=rand(3), γn=rand(3)), Binary(; θ=randn(2)), randn(3,2),
        randn(3), randn(2),
    )
    v = randn(3, 100)
    vdims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    gs = gradient(rbm) do rbm
        F = mean(free_energy(rbm, v))
        urbm = uncenter(rbm)
        L2_fields = sum(abs2, urbm.visible.θp) + sum(abs2, urbm.visible.θn)
        L1_weights = sum(abs, urbm.w)
        L2_weights = sum(abs2, urbm.w)
        L2L1_weights = sum(abs2, sum(abs, urbm.w; dims=vdims))
        return (
            F + l2_fields/2 * L2_fields +
            l1_weights * L1_weights +
            l2_weights/2 * L2_weights +
            l2l1_weights/(2N) * L2L1_weights
        )
    end

    ∂ = ∂free_energy(rbm, v)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
    @test only(gs).w ≈ ∂.w
end
