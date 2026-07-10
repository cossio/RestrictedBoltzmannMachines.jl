# Tests GPU compatibility without a physical GPU, using JLArrays (the reference
# GPUArrays.jl backend). With allowscalar(false), any code path falling back to
# scalar indexing errors out, just like CuArray on CI. The setting is session-global,
# which is intentional: all JLArrays tests live in this file, and any future test
# using JLArrays should run under allowscalar(false) too.
import Random
using Test: @test, @testset, @test_broken
using Statistics: mean
using Random: bitrand
using Adapt: adapt
using JLArrays: JLArray, JLArrays
using RestrictedBoltzmannMachines: RBM, CenteredRBM, StandardizedRBM, ∂RBM, BinaryRBM,
    Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU, nsReLU, PottsGumbel,
    energy, energies, free_energy, cgfs, ∂cgfs,
    sample_from_inputs, mean_from_inputs, var_from_inputs, std_from_inputs,
    mode_from_inputs, meanvar_from_inputs,
    inputs_h_from_v, inputs_v_from_h, mean_h_from_v, mean_v_from_h,
    sample_h_from_v, sample_v_from_h, sample_v_from_v, reconstruction_error,
    log_pseudolikelihood, log_pseudolikelihood_stoch,
    initialize!, pcd!, ∂free_energy, zerosum!, rescale_weights!, aise, raise

JLArrays.allowscalar(false)

# CUDA.jl provides the RNG-less `Random.rand!(::AnyCuArray)`, but JLArrays does not, and
# the Random stdlib's `rand!(A)` skips the two-arg method that GPUArrays implements
# GPU-friendly, falling into a scalar-indexing loop. Emulate the CUDA.jl overload here.
Random.rand!(A::JLArray) = Random.rand!(Random.default_rng(), A)

const N = (3, 2) # visible site grid
const Q = 4 # Potts classes
const B = 16 # batch size

random_layers() = (
    Binary(; θ = randn(N...)),
    Spin(; θ = randn(N...)),
    Potts(; θ = randn(Q, N...)),
    Gaussian(; θ = randn(N...), γ = 1 .+ rand(N...)),
    ReLU(; θ = randn(N...), γ = 1 .+ rand(N...)),
    dReLU(; θp = randn(N...), θn = randn(N...), γp = 1 .+ rand(N...), γn = 1 .+ rand(N...)),
    pReLU(; θ = randn(N...), γ = 1 .+ rand(N...), Δ = randn(N...), η = rand(N...) .- 0.5),
    xReLU(; θ = randn(N...), γ = 1 .+ rand(N...), Δ = randn(N...), ξ = randn(N...)),
    nsReLU(; θ = randn(N...), Δ = randn(N...), ξ = randn(N...)),
    PottsGumbel(; θ = randn(Q, N...)),
)

@testset "adapt round-trip" begin
    for layer in random_layers()
        jl_layer = adapt(JLArray, layer)
        @test jl_layer isa typeof(layer).name.wrapper
        @test jl_layer.par isa JLArray
        @test size(jl_layer) == size(layer)
        @test adapt(Array, jl_layer).par == layer.par
    end

    rbm = BinaryRBM(randn(N...), randn(3), randn(N..., 3))
    jl_rbm = adapt(JLArray, rbm)
    @test jl_rbm.visible.par isa JLArray
    @test jl_rbm.hidden.par isa JLArray
    @test jl_rbm.w isa JLArray
    @test adapt(Array, jl_rbm).w == rbm.w

    centered = CenteredRBM(rbm.visible, rbm.hidden, rbm.w, randn(N...), randn(3))
    jl_centered = adapt(JLArray, centered)
    @test jl_centered.offset_v isa JLArray
    @test jl_centered.offset_h isa JLArray

    standardized = StandardizedRBM(
        rbm.visible, rbm.hidden, rbm.w,
        randn(N...), randn(3), 1 .+ rand(N...), 1 .+ rand(3)
    )
    jl_standardized = adapt(JLArray, standardized)
    @test jl_standardized.scale_v isa JLArray
    @test jl_standardized.scale_h isa JLArray

    ∂ = ∂free_energy(rbm, falses(N..., B))
    jl_∂ = adapt(JLArray, ∂)
    @test jl_∂.w isa JLArray
    @test adapt(Array, jl_∂.w) == ∂.w
end

@testset "layer functions: $(typeof(layer).name.wrapper)" for layer in random_layers()
    jl_layer = adapt(JLArray, layer)
    inputs = randn(size(layer)..., B)
    jl_inputs = JLArray(inputs)

    @test adapt(Array, cgfs(jl_layer, jl_inputs)) ≈ cgfs(layer, inputs)
    @test adapt(Array, mean_from_inputs(jl_layer, jl_inputs)) ≈ mean_from_inputs(layer, inputs)
    @test adapt(Array, var_from_inputs(jl_layer, jl_inputs)) ≈ var_from_inputs(layer, inputs)
    @test adapt(Array, std_from_inputs(jl_layer, jl_inputs)) ≈ std_from_inputs(layer, inputs)
    @test adapt(Array, mode_from_inputs(jl_layer, jl_inputs)) == mode_from_inputs(layer, inputs)
    μ, ν = meanvar_from_inputs(jl_layer, jl_inputs)
    μ_cpu, ν_cpu = meanvar_from_inputs(layer, inputs)
    @test adapt(Array, μ) ≈ μ_cpu
    @test adapt(Array, ν) ≈ ν_cpu
    @test adapt(Array, ∂cgfs(jl_layer, jl_inputs)) ≈ ∂cgfs(layer, inputs)

    x = sample_from_inputs(layer, inputs)
    jl_x = JLArray(float(x))
    @test adapt(Array, energies(jl_layer, jl_x)) ≈ energies(layer, float(x))
    @test adapt(Array, energy(jl_layer, jl_x)) ≈ energy(layer, float(x))

    if layer isa Potts
        # Potts sampling is documented as not GPU-friendly (uses scalar indexing);
        # PottsGumbel is the GPU alternative.
        @test_broken sample_from_inputs(jl_layer, jl_inputs) isa JLArray
    else
        s = sample_from_inputs(jl_layer, jl_inputs)
        @test s isa JLArray
        @test size(s) == (size(layer)..., B)
        @test all(isfinite, adapt(Array, s))
    end
end

@testset "RBM functions" begin
    rbm = BinaryRBM(randn(N...), randn(3), randn(N..., 3) / √prod(N))
    jl_rbm = adapt(JLArray, rbm)
    v = float(bitrand(N..., B))
    h = float(bitrand(3, B))
    jl_v = JLArray(v)
    jl_h = JLArray(h)

    @test adapt(Array, inputs_h_from_v(jl_rbm, jl_v)) ≈ inputs_h_from_v(rbm, v)
    @test adapt(Array, inputs_v_from_h(jl_rbm, jl_h)) ≈ inputs_v_from_h(rbm, h)
    @test adapt(Array, energy(jl_rbm, jl_v, jl_h)) ≈ energy(rbm, v, h)
    @test adapt(Array, free_energy(jl_rbm, jl_v)) ≈ free_energy(rbm, v)
    @test adapt(Array, mean_h_from_v(jl_rbm, jl_v)) ≈ mean_h_from_v(rbm, v)
    @test adapt(Array, mean_v_from_h(jl_rbm, jl_h)) ≈ mean_v_from_h(rbm, h)

    @test sample_h_from_v(jl_rbm, jl_v) isa JLArray
    @test sample_v_from_h(jl_rbm, jl_h) isa JLArray
    jl_v_sampled = sample_v_from_v(jl_rbm, jl_v; steps = 3)
    @test jl_v_sampled isa JLArray
    @test size(jl_v_sampled) == size(jl_v)
    @test all(adapt(Array, reconstruction_error(jl_rbm, jl_v)) .≥ 0)

    ∂ = ∂free_energy(jl_rbm, jl_v)
    @test ∂.w isa JLArray
    @test adapt(Array, ∂.w) ≈ ∂free_energy(rbm, v).w
end

@testset "pseudolikelihood" begin
    # Binary and Potts visible layers (Potts is the typical use case)
    rbm_binary = BinaryRBM(randn(N...), randn(3), randn(N..., 3) / √prod(N))
    v_binary = float(bitrand(N..., B))

    potts_visible = Potts(; θ = randn(Q, N...))
    w = randn(Q, N..., 3) / √(Q * prod(N))
    rbm_potts = RBM(potts_visible, Binary(; θ = randn(3)), w)
    v_potts = float(sample_from_inputs(potts_visible, zeros(Q, N..., B)))

    for (rbm, v) in ((rbm_binary, v_binary), (rbm_potts, v_potts))
        jl_rbm = adapt(JLArray, rbm)
        jl_v = JLArray(v)
        # exact pseudolikelihood is deterministic: must match CPU
        @test adapt(Array, log_pseudolikelihood(jl_rbm, jl_v; exact = true)) ≈
            log_pseudolikelihood(rbm, v; exact = true)
        # stochastic version: smoke test (random site selection)
        lpl = log_pseudolikelihood_stoch(jl_rbm, jl_v)
        @test all(isfinite, adapt(Array, lpl))
        @test length(lpl) == B
    end
end

@testset "gauge transformations" begin
    rbm = RBM(
        Potts(; θ = randn(Q, N...)),
        ReLU(; θ = randn(3), γ = 1 .+ rand(3)),
        randn(Q, N..., 3) / √(Q * prod(N)),
    )
    jl_rbm = adapt(JLArray, rbm)

    zerosum!(jl_rbm)
    zerosum!(rbm)
    @test adapt(Array, jl_rbm.w) ≈ rbm.w
    @test adapt(Array, jl_rbm.visible.par) ≈ rbm.visible.par

    rescale_weights!(jl_rbm)
    rescale_weights!(rbm)
    @test adapt(Array, jl_rbm.w) ≈ rbm.w
    @test adapt(Array, jl_rbm.hidden.par) ≈ rbm.hidden.par

    # zerosum! on StandardizedRBM with nontrivial offsets/scales
    std_rbm = StandardizedRBM(
        Potts(; θ = randn(Q, N...)),
        ReLU(; θ = randn(3), γ = 1 .+ rand(3)),
        randn(Q, N..., 3) / √(Q * prod(N)),
        randn(Q, N...) / 3, randn(3) / 3,
        1 .+ rand(Q, N...), 1 .+ rand(3),
    )
    jl_std_rbm = adapt(JLArray, std_rbm)
    zerosum!(jl_std_rbm)
    zerosum!(std_rbm)
    @test adapt(Array, jl_std_rbm.w) ≈ std_rbm.w
    @test adapt(Array, jl_std_rbm.visible.par) ≈ std_rbm.visible.par
    @test adapt(Array, jl_std_rbm.hidden.par) ≈ std_rbm.hidden.par
end

@testset "initialize! and pcd!" begin
    data = float(bitrand(N..., 512))
    jl_data = JLArray(data)

    rbm = BinaryRBM(zeros(N...), zeros(2), zeros(N..., 2))
    jl_rbm = adapt(JLArray, rbm)
    initialize!(jl_rbm, jl_data)
    initialize!(rbm, data)
    @test adapt(Array, jl_rbm.visible.par) ≈ rbm.visible.par

    pcd!(jl_rbm, jl_data; iters = 10, batchsize = 32, steps = 2)
    @test all(isfinite, adapt(Array, jl_rbm.w))
    @test all(isfinite, adapt(Array, jl_rbm.visible.par))
    @test all(isfinite, adapt(Array, jl_rbm.hidden.par))
end

@testset "AIS" begin
    rbm = BinaryRBM(randn(N...), randn(2), randn(N..., 2) / √prod(N))
    jl_rbm = adapt(JLArray, rbm)
    R = aise(jl_rbm; nbetas = 20, nsamples = 4)
    @test all(isfinite, adapt(Array, R))
    jl_v = JLArray(float(bitrand(N..., 4)))
    R = raise(jl_rbm; nbetas = 20, v = jl_v)
    @test all(isfinite, adapt(Array, R))
end
