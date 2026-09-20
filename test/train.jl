#= Moment-matching tests for the training algorithms.

At a maximum-likelihood stationary point, the model must match the data moments
of the sufficient statistics: ⟨v⟩, ⟨h⟩ (with h averaged over p(h|v)), and ⟨v ⊗ h⟩.
For small machines the model moments can be computed exactly by enumerating all
visible states, so well-trainedness can be checked without sampling noise on
tiny synthetic datasets. =#

using Test: @test, @testset
using Statistics: mean
using LinearAlgebra: Diagonal, norm
using Random: seed!
using LogExpFunctions: softmax
using Optimisers: Adam, Descent
using RestrictedBoltzmannMachines: RBM, BinaryRBM, Binary, Spin, Potts, Gaussian,
    pcd!, initialize!, free_energy, log_likelihood, log_partition,
    collect_states, mean_h_from_v, generate_sequences, onehot_encode,
    center, uncenter, standardize, unstandardize, weight_norms,
    CenteredRBM, StandardizedRBM

all_visible_states(layer::Union{Binary, Spin}) = collect_states(layer)

function all_visible_states(layer::Potts)
    Q, N = size(layer)
    states = [onehot_encode(s, 1:Q) for s in generate_sequences(N, 1:Q)]
    return cat(states...; dims = 3)
end

flatten_batch(x::AbstractArray) = Float64.(reshape(x, :, size(x)[end]))

#= Largest absolute moment-matching residuals between the exact model
distribution (by enumeration of visible states) and the (weighted) data. =#
function moment_gaps(rbm, data; wts = nothing)
    states = all_visible_states(rbm.visible)
    p = softmax(-free_energy(rbm, states))
    vmod = flatten_batch(states)
    hmod = flatten_batch(mean_h_from_v(rbm, states))
    vdat = flatten_batch(data)
    hdat = flatten_batch(mean_h_from_v(rbm, data))
    nsamples = size(data)[end]
    q = isnothing(wts) ? fill(1 / nsamples, nsamples) : wts ./ sum(wts)
    return (;
        v = maximum(abs, vmod * p - vdat * q),
        h = maximum(abs, hmod * p - hdat * q),
        vh = maximum(abs, vmod * Diagonal(p) * hmod' - vdat * Diagonal(q) * hdat'),
    )
end

plain_rbm(rbm::RBM) = rbm
plain_rbm(rbm::CenteredRBM) = uncenter(rbm)
plain_rbm(rbm::StandardizedRBM) = unstandardize(rbm)

#= Trains `rbm` in place with `pcd!` and returns a plain `RBM` whose parameters are
the average of the (equivalent plain RBM) iterates over the last `tail` updates.

With a constant learning rate the final iterate of stochastic training jitters around
the stationary point by an amount set by the learning rate and the minibatch noise,
not by how well the model was trained; and some models amplify that jitter in the
moments (a Gaussian hidden unit with small γ has ⟨h⟩ ∝ 1/γ). Checking moment
matching on the final iterate therefore depends on where in that jitter a given RNG
stream leaves it, and a change of the RNG stream (Julia 1.13 changed the seeded
output of `shuffle`) was enough to push seeded cases over their thresholds. The tail
average is a far less noisy estimate of the stationary point, provided training has
converged before the tail starts: with Adam(1e-3) the binary and Potts cases below
are still drifting at 5000 updates, hence 10000. The gauge assertions below are
still made on the trained `rbm` itself, since they concern `pcd!`. =#
function pcd_averaged!(rbm, data; iters::Int, tail::Int = iters ÷ 5, kwargs...)
    avg = nothing
    n = 0
    function callback(; rbm, iter, _...)
        iter > iters - tail || return nothing
        plain = plain_rbm(rbm)
        if isnothing(avg)
            avg = deepcopy(plain)
        else
            avg.visible.par .+= plain.visible.par
            avg.hidden.par .+= plain.hidden.par
            avg.w .+= plain.w
        end
        n += 1
        return nothing
    end
    pcd!(rbm, data; iters, kwargs..., callback)
    avg.visible.par ./= n
    avg.hidden.par ./= n
    avg.w ./= n
    return avg
end

# 3 correlated binary units, with non-uniform means and pairwise correlations
function binary_dataset()
    patterns = Bool[
        1 1 0 0 1
        1 1 0 1 0
        1 0 0 1 1
    ]
    counts = [30, 20, 30, 10, 10]
    return reduce(hcat, [repeat(patterns[:, k], 1, counts[k]) for k in eachindex(counts)])
end

spin_dataset() = Int8.(2 .* binary_dataset() .- 1)

# 2 correlated Potts units with Q = 3 classes
function potts_dataset()
    patterns = [
        1 2 3 1 2
        1 2 3 3 1
    ]
    counts = [35, 25, 20, 10, 10]
    classes = reduce(hcat, [repeat(patterns[:, k], 1, counts[k]) for k in eachindex(counts)])
    return onehot_encode(classes, 1:3)
end

@testset "pcd binary moment matching: $name" for (name, kwargs) in [
        ("Adam", (; batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3))),
        # 1-step chains fall behind the sharpening model over longer runs, so 5000 updates
        ("steps=1, no shuffle", (; batchsize = 16, iters = 5000, steps = 1, shuffle = false, optim = Adam(1.0e-3))),
        ("SGD", (; batchsize = 32, iters = 10000, steps = 5, optim = Descent(2.0e-2))),
    ]
    seed!(7)
    data = binary_dataset()
    rbm = BinaryRBM(3, 5)
    initialize!(rbm, data)
    avg = pcd_averaged!(rbm, data; kwargs...)
    gaps = moment_gaps(avg, data)
    @test gaps.v < 0.05
    @test gaps.h < 0.05
    @test gaps.vh < 0.05
end

@testset "pcd improves exact log-likelihood" begin
    seed!(11)
    data = binary_dataset()
    rbm = BinaryRBM(3, 5)
    initialize!(rbm, data)
    ll_init = mean(log_likelihood(rbm, data))
    pcd!(rbm, data; batchsize = 32, iters = 5000, steps = 5, optim = Adam(1.0e-3))
    ll_trained = mean(log_likelihood(rbm, data))
    @test ll_trained > ll_init

    # compare against the entropy bound: ll ≤ mean(log(empirical probability))
    counts = Dict{Vector{Bool}, Int}()
    for v in eachcol(data)
        counts[v] = get(counts, v, 0) + 1
    end
    ll_bound = sum(c * log(c / size(data, 2)) for c in values(counts)) / size(data, 2)
    @test ll_trained ≤ ll_bound + 1.0e-3
    @test ll_trained > ll_bound - 0.25 # close to the optimum, up to model misspecification
end

@testset "pcd with weighted data" begin
    seed!(23)
    data = binary_dataset()
    wts = [v[1] ? 3.0 : 1.0 for v in eachcol(data)]
    rbm = BinaryRBM(3, 5)
    initialize!(rbm, data; wts)
    avg = pcd_averaged!(rbm, data; wts, batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3))
    gaps = moment_gaps(avg, data; wts)
    @test gaps.v < 0.05
    @test gaps.h < 0.05
    @test gaps.vh < 0.05
end

@testset "pcd with weight regularization" begin
    seed!(31)
    data = binary_dataset()

    rbm_weak = BinaryRBM(3, 5)
    initialize!(rbm_weak, data)
    avg_weak = pcd_averaged!(rbm_weak, data; batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3), l2_weights = 1.0e-4)

    rbm_strong = BinaryRBM(3, 5)
    initialize!(rbm_strong, data)
    avg_strong = pcd_averaged!(rbm_strong, data; batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3), l2_weights = 10.0)

    # weak regularization barely perturbs the maximum-likelihood solution
    gaps = moment_gaps(avg_weak, data)
    @test gaps.v < 0.05
    @test gaps.vh < 0.1

    # strong weight decay kills the weights, but fields are unregularized,
    # so single-site statistics are still matched
    @test norm(rbm_strong.w) < norm(rbm_weak.w) / 10
    @test moment_gaps(avg_strong, data).v < 0.05
end

@testset "pcd spin moment matching" begin
    seed!(43)
    data = spin_dataset()
    rbm = RBM(Spin((3,)), Spin((4,)), zeros(3, 4))
    initialize!(rbm, data)
    avg = pcd_averaged!(rbm, data; batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3))
    gaps = moment_gaps(avg, data)
    @test gaps.v < 0.1 # spins range over [-1, 1]
    @test gaps.h < 0.1
    @test gaps.vh < 0.1
end

@testset "pcd potts moment matching" begin
    seed!(47)
    data = potts_dataset()
    rbm = RBM(Potts((3, 2)), Binary((3,)), zeros(3, 2, 3))
    initialize!(rbm, data)
    avg = pcd_averaged!(rbm, data; batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3))
    gaps = moment_gaps(avg, data)
    @test gaps.v < 0.05
    @test gaps.h < 0.05
    @test gaps.vh < 0.05
    # zerosum gauge is maintained
    @test norm(mean(rbm.visible.θ; dims = 1)) < 1.0e-10
    @test norm(mean(rbm.w; dims = 1)) < 1.0e-10
end

@testset "pcd potts learns visible fields" begin
    #= pcd! with zerosum=true (the default) must still learn the Potts visible
    fields: the zerosum projection of the gradient must not discard them. The
    moment-matching tests above are insensitive to this because initialize!
    already sets the fields near their maximum-likelihood values. Here training
    starts from zero fields on data with non-uniform color frequencies, so it
    must move the fields, while staying in the zerosum gauge. =#
    seed!(69)
    data = potts_dataset()
    rbm = RBM(Potts((3, 2)), Binary((3,)), zeros(3, 2, 3))
    pcd!(rbm, data; batchsize = 32, iters = 100, steps = 5, optim = Descent(1.0e-2))
    @test norm(rbm.visible.θ) > 1.0e-3
    # while staying in the zerosum gauge
    @test norm(mean(rbm.visible.θ; dims = 1)) < 1.0e-10
end

@testset "pcd gaussian hidden moment matching" begin
    seed!(53)
    data = binary_dataset()
    rbm = RBM(Binary((3,)), Gaussian((2,)), zeros(3, 2))
    initialize!(rbm, data)
    avg = pcd_averaged!(rbm, data; batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3))
    gaps = moment_gaps(avg, data)
    @test gaps.v < 0.05
    @test gaps.h < 0.1
    @test gaps.vh < 0.1
    # rescale gauge: unit weight norm per hidden unit
    @test weight_norms(rbm) ≈ ones(2)
end

@testset "centered pcd moment matching" begin
    seed!(59)
    data = binary_dataset()
    rbm = center(BinaryRBM(3, 5))
    initialize!(rbm, data)
    avg = pcd_averaged!(rbm, data; batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3))
    gaps = moment_gaps(avg, data)
    @test gaps.v < 0.05
    @test gaps.h < 0.05
    @test gaps.vh < 0.05
end

@testset "standardized pcd moment matching" begin
    seed!(61)
    data = binary_dataset()
    rbm = standardize(initialize!(BinaryRBM(3, 5), data))
    avg = pcd_averaged!(rbm, data; batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3))
    gaps = moment_gaps(avg, data)
    @test gaps.v < 0.05
    @test gaps.h < 0.05
    @test gaps.vh < 0.05
end

@testset "standardized pcd with weighted data" begin
    seed!(62)
    data = binary_dataset()
    wts = [v[1] ? 3.0 : 1.0 for v in eachcol(data)]
    rbm = standardize(initialize!(BinaryRBM(3, 5), data; wts))
    avg = pcd_averaged!(rbm, data; wts, batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3))
    gaps = moment_gaps(avg, data; wts)
    @test gaps.v < 0.05
    @test gaps.h < 0.05
    @test gaps.vh < 0.05
end

@testset "standardized pcd potts moment matching" begin
    seed!(63)
    data = potts_dataset()
    rbm = standardize(initialize!(RBM(Potts((3, 2)), Binary((3,)), zeros(3, 2, 3)), data))
    avg = pcd_averaged!(rbm, data; batchsize = 32, iters = 10000, steps = 5, optim = Adam(1.0e-3))
    gaps = moment_gaps(avg, data)
    @test gaps.v < 0.05
    @test gaps.h < 0.05
    @test gaps.vh < 0.05
    # zerosum gauge of the equivalent unstandardized RBM is maintained
    urbm = unstandardize(rbm)
    @test norm(mean(urbm.visible.θ; dims = 1)) < 1.0e-10
    @test norm(mean(urbm.w; dims = 1)) < 1.0e-10
end
