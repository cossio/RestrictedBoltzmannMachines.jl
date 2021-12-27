using RestrictedBoltzmannMachines: ExpDecay
using Flux: Optimise
using Random, Flux

@testset "ExpDecay sanity check" begin
    o = ExpDecay(0.2, 0.5, 1, 1e-3)
    p = [0.0]
    steps = 1:8
    eta_expected = @. max(o.eta * 0.5 ^ steps, o.clip)
    eta_actual = [Optimise.apply!(o, p, [1.0])[1] for _ in steps]
    @test eta_actual == eta_expected
end

@testset "ExpDecay with starting step" begin
    start = 4
    o = ExpDecay(0.2, 0.5, 1, 1e-3, start)
    p = [0.0]
    steps = 1:8
    eta_expected = @. max(o.eta * 0.5 ^ max(steps - start + 1, 0), o.clip)
    eta_actual = [Optimise.apply!(o, p, [1.0])[1] for _ in steps]
    @test eta_actual == eta_expected
end

w = randn(10, 10)
o = ExpDecay(0.1, 0.1, 1000, 1e-4)
w1 = randn(10,10)
loss(x) = Flux.Losses.mse(w*x, w1*x)
flag = 1
decay_steps = []
for t = 1:10^5
    prev_eta = o.eta
    θ = Flux.Params([w1])
    x = rand(10)
    θ̄ = gradient(() -> loss(x), θ)
    prev_grad = collect(θ̄[w1])
    delta = Optimise.apply!(o, w1, θ̄[w1])
    w1 .-= delta
    new_eta = o.eta
    if new_eta != prev_eta
        push!(decay_steps, t)
    end
    array = fill(o.eta, size(prev_grad))
    @test array .* prev_grad == delta
    # if array .* prev_grad != delta
    #     flag = 0
    # end
end
#@test flag == 1
# Test to check if decay happens at decay steps. Eta reaches clip value (1e-4) after 4000 steps (decay by 0.1 every 1000 steps starting at 0.1).
ground_truth = []
for i in 1:4
    push!(ground_truth, 1000*i)  # Expected decay steps for this example.
end
@test decay_steps == ground_truth
@test o.eta == o.clip


@testset "default_optimizer" begin

end
