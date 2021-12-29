include("tests_init.jl")

@testset "ExpDecay sanity check" begin
    o = RBMs.ExpDecay(0.2, 0.5, 1, 1e-3)
    p = [0.0]
    steps = 1:8
    eta_expected = @. max(o.eta * 0.5 ^ steps, o.clip)
    eta_actual = [Flux.Optimise.apply!(o, p, [1.0])[1] for _ in steps]
    @test eta_actual == eta_expected
end

@testset "ExpDecay with starting step" begin
    start = 4
    o = RBMs.ExpDecay(0.2, 0.5, 1, 1e-3, start)
    p = [0.0]
    steps = 1:8
    eta_expected = @. max(o.eta * 0.5 ^ max(steps - start, 0), o.clip)
    eta_actual = [Flux.Optimise.apply!(o, p, [1.0])[1] for _ in steps]
    @test eta_actual == eta_expected
end

w = randn(10, 10)
o = RBMs.ExpDecay(0.1, 0.1, 1000, 1e-4)
w1 = randn(10,10)
loss(x) = Flux.Losses.mse(w*x, w1*x)
flag = 1
decay_steps = []
for t = 1:10^5
    prev_eta = o.eta
    θ = Flux.Params([w1])
    x = rand(10)
    θ̄ = Zygote.gradient(() -> loss(x), θ)
    prev_grad = collect(θ̄[w1])
    delta = Flux.Optimise.apply!(o, w1, θ̄[w1])
    w1 .-= delta
    new_eta = o.eta
    if new_eta != prev_eta
        push!(decay_steps, t)
    end
    array = fill(o.eta, size(prev_grad))
    @test array .* prev_grad == delta
end

# Test to check if decay happens at decay steps. Eta reaches clip value (1e-4) after 4000 steps (decay by 0.1 every 1000 steps starting at 0.1).
ground_truth = []
for i in 1:4
    push!(ground_truth, 1000*i)  # Expected decay steps for this example.
end
@test decay_steps == ground_truth
@test o.eta == o.clip


@testset "default_optimizer" begin
    nsamples = 60000
    batchsize = 128
    epochs = 100

    o = RBMs.default_optimizer(nsamples, batchsize, epochs; opt=Flux.Descent(1), decay_final=0.01, decay_after=0.5)
    p = [0.0]
    lrs = [only(Flux.Optimise.apply!(o, p, [1.0])) for b in RBMs.minibatches(nsamples; batchsize=batchsize) for epoch in 1:epochs]

    steps_per_epoch = RBMs.minibatch_count(nsamples; batchsize = batchsize)
    nsteps = steps_per_epoch * epochs
    start = round(Int, nsteps * 0.5)
    lrs_expected = [0.01^((max(n - start, 0) ÷ steps_per_epoch) / (max(nsteps - start, 0) ÷ steps_per_epoch)) for n in 1:nsteps]

    @test lrs ≈ lrs_expected
end


@testset "ADAM" begin

    @testset "Optimise" begin
        # Ensure rng has different state inside and outside the inner @testset
        # so that w and w' are different
        Random.seed!(84)
        w = randn(10, 10)
        @testset for opt in [RBMs.ADAM()]
            Random.seed!(42)
            w′ = randn(10, 10)
            b = Flux.Zeros()
            loss(x) = Flux.Losses.mse(w*x, w′*x .+ b)
            for t = 1: 10^5
                θ = Flux.params([w′, b])
                x = rand(10)
                θ̄ = Zygote.gradient(() -> loss(x), θ)
                Flux.Optimise.update!(opt, θ, θ̄)
            end
            @test loss(rand(10, 10)) < 0.01
        end
    end

    @testset "Momentum Optimisers and complex values" begin
        # Test every optimizer that has momentum internally
        for opt_ctor in [RBMs.ADAM]
            # Our "model" is just a complex number
            w = zeros(ComplexF32, 1)

            # Our model attempts to learn `f(x) = conj(x)` where `f(x) = w*x`
            function loss()
                # Deterministic training data is the best training data
                x = ones(1, 1) + 1im*ones(1, 1)

                # Manually implement `mse()` to allow demonstration of brokenness
                # on older Flux builds that don't have a fixed `mse()`
                return sum(abs2.(w * x .- conj(x)))
            end

            params = Flux.Params([w])
            opt = opt_ctor(1e-2)

            # Train for 10 iterations, enforcing that loss is monotonically decreasing
            last_loss = Inf
            for idx in 1:10
                grads = Flux.gradient(loss, params)
                @test loss() < last_loss
                last_loss = loss()
                Flux.update!(opt, params, grads)
            end
        end
    end
end

@testset "SqrtDecay" begin
    Random.seed!(84)
    w = randn(10, 10)
    w_ = randn(10, 10)
    loss(x) = Flux.Losses.mse(w * x, w_ * x)
    opt = Flux.Optimiser(RBMs.SqrtDecay(; decay=5), Flux.ADAM(0.001))
    for t = 1:10^5
        θ = Zygote.Params([w_])
        x = rand(10)
        θ_ = Zygote.gradient(() -> loss(x), θ)
        Flux.Optimise.update!(opt, θ, θ_)
    end
    @test loss(rand(10, 10)) < 0.01
end
