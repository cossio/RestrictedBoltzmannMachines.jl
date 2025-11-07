import Revise
using RestrictedBoltzmannMachines: BinaryRBM, interaction_energy
using BenchmarkTools: @btime
using Random: bitrand, randn!
using Test: @test, @inferred

N = 500
M = 2
T = 1000

rbm = BinaryRBM(N, M)
randn!(rbm.w); randn!(rbm.visible.θ); randn!(rbm.hidden.θ);

inter1(rbm, v, h) = interaction_energy(rbm, v, h)
inter2(rbm, v, h) = -v' * rbm.w * h

function foo()
    rbm = BinaryRBM(N, M)
    randn!(rbm.w); randn!(rbm.visible.θ); randn!(rbm.hidden.θ);
    v = bitrand(N, T)
    h = bitrand(M)
    results = zeros(Float64, 10000000)
    for t = 1:5
        results[t] = sum(inter1(rbm, v, h))
    end
    return results
end

println(" --- now with rand(Bool) --- ")

v = rand(Bool, N, T)
h = rand(Bool, M)
@test inter1(rbm, v, h) ≈ inter2(rbm, v, h)
@inferred inter1(rbm, v, h)
@inferred inter2(rbm, v, h)

@btime inter1($rbm, $v, $h);
@btime inter2($rbm, $v, $h);

println(" --- now with bitrand --- ")

v = bitrand(N, T)
h = bitrand(M)
@test inter1(rbm, v, h) ≈ inter2(rbm, v, h)
@inferred inter1(rbm, v, h)
@inferred inter2(rbm, v, h)

@btime inter1($rbm, $v, $h);
@btime inter2($rbm, $v, $h);

println(" --- now with float --- ")

v = float(bitrand(N, T))
h = float(bitrand(M))
@test inter1(rbm, v, h) ≈ inter2(rbm, v, h)
@inferred inter1(rbm, v, h)
@inferred inter2(rbm, v, h)

@btime inter1($rbm, $v, $h);
@btime inter2($rbm, $v, $h);

println("done")
