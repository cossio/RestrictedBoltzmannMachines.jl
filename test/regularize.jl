import RestrictedBoltzmannMachines as RBMs
import Zygote
using Test: @test, @testset
using Statistics: mean
using Random: bitrand
using RestrictedBoltzmannMachines: BinaryRBM, visible, hidden, weights
using RestrictedBoltzmannMachines: free_energy, ∂free_energy, ∂regularize!

@testset "regularization" begin
    rbm = BinaryRBM(randn(3,5), randn(3,2), randn(3,5,3,2))
    v = bitrand(3,5,100)
    vdims = ntuple(identity, ndims(visible(rbm)))
    N = length(visible(rbm))

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    gs = Zygote.gradient(rbm) do rbm
        F = mean(free_energy(rbm, v))
        L2_fields = sum(abs2, visible(rbm).θ)
        L1_weights = sum(abs, weights(rbm))
        L2_weights = sum(abs2, weights(rbm))
        L2L1_weights = sum(abs2, sum(abs, weights(rbm); dims=vdims))
        return (
            F + l2_fields/2 * L2_fields + l1_weights * L1_weights +
            l2_weights/2 * L2_weights +
            l2l1_weights/(2N) * L2L1_weights
        )
    end

    ∂ = ∂free_energy(rbm, v)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

    @test only(gs).visible.θ ≈ ∂.visible.θ
    @test only(gs).hidden.θ ≈ ∂.hidden.θ
    @test only(gs).w ≈ ∂.w
end
