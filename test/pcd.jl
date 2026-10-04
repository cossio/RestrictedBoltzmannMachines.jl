import Random
using Test: @test, @testset
using Statistics: mean
using Random: bitrand
using Optimisers: Adam
using RestrictedBoltzmannMachines: BinaryRBM, HopfieldRBM, sample_v_from_v, initialize!, pcd!, CenteredRBM
using RestrictedBoltzmannMachines: RBM, Potts, Gaussian, StandardizedRBM, inputs_h_from_v, free_energy
using RestrictedBoltzmannMachines: standardize_visible_from_data!, standardize_hidden_from_v!, unstandardize, standardize
using RestrictedBoltzmannMachines: PlainStandardizedRBM
using FillArrays: Falses, Trues

Random.seed!(23)

@testset "pcd ($name)" for (name, wrap) in (("plain", identity), ("centered", CenteredRBM))
    data = falses(2, 1000)
    data[1, 1:2:end] .= true
    data[2, 1:2:end] .= true

    rbm = wrap(BinaryRBM(2, 5))
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

@testset "pcd! ps, state, and unified callback keywords ($name)" for (name, wrap) in (("plain", identity), ("centered", CenteredRBM))
    data = bitrand(2, 32)
    rbm = wrap(BinaryRBM(2, 3))
    seen = Ref{Any}(nothing)
    state, ps = pcd!(
        rbm, data;
        iters = 2, batchsize = 8,
        callback = (; kwargs...) -> (seen[] = kwargs),
    )
    @test issubset((:rbm, :optim, :state, :ps, :iter, :vd, :wd, :∂, :vm), keys(seen[]))
    @test seen[][:rbm] === rbm
    @test seen[][:ps] === ps
    @test seen[][:state] == state

    # training can be continued from the returned optimizer state and parameters
    state2, ps2 = pcd!(rbm, data; iters = 2, batchsize = 8, ps, state)
    @test ps2 === ps
    @test all(isfinite, rbm.visible.par)
    @test all(isfinite, rbm.hidden.par)
    @test all(isfinite, rbm.w)
end

@testset "plain RBM trains as a StandardizedRBM with fixed offsets and scales" begin
    rbm = RBM(Potts((3, 2)), Gaussian((2,)), randn(Float32, 3, 2, 2))
    std_rbm = PlainStandardizedRBM(rbm)
    @test std_rbm isa StandardizedRBM && std_rbm isa CenteredRBM
    @test std_rbm.visible === rbm.visible && std_rbm.hidden === rbm.hidden && std_rbm.w === rbm.w
    @test std_rbm.offset_v isa Falses && std_rbm.offset_h isa Falses
    @test std_rbm.scale_v isa Trues && std_rbm.scale_h isa Trues
    @test unstandardize(std_rbm) isa RBM
    @test standardize(StandardizedRBM(rbm)) isa PlainStandardizedRBM # lazy Zeros and Ones

    # Bool (one-hot) data: no promotion, and the same values as the plain RBM
    data = falses(3, 2, 8)
    for n in 1:8, i in 1:2
        data[rand(1:3), i, n] = true
    end
    @test inputs_h_from_v(std_rbm, data) == inputs_h_from_v(rbm, data)
    @test eltype(inputs_h_from_v(std_rbm, data)) === eltype(inputs_h_from_v(rbm, data)) === Float32
    @test free_energy(std_rbm, data) == free_energy(rbm, data)

    # fitting statistics from data changes nothing
    pars = (copy(rbm.visible.par), copy(rbm.hidden.par), copy(rbm.w))
    standardize_visible_from_data!(std_rbm, data)
    standardize_hidden_from_v!(std_rbm, data)
    @test (rbm.visible.par, rbm.hidden.par, rbm.w) == pars

    # the standardization keywords are accepted (and irrelevant) for a plain RBM; the
    # callback receives the plain RBM
    seen = Ref{Any}(nothing)
    pcd!(
        rbm, data; iters = 2, batchsize = 4, l2_weights = 0.1, damping = 0.5, ϵv = 0.1, ϵh = 0.1,
        regularize_unstandardized = false, callback = (; rbm, _...) -> (seen[] = rbm)
    )
    @test seen[] === rbm
    @test all(isfinite, rbm.visible.par)
    @test all(isfinite, rbm.hidden.par)
    @test all(isfinite, rbm.w)
end
