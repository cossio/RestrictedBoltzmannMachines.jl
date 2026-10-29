using Test: @test, @testset, @inferred
using SpecialFunctions: erfcx
import RestrictedBoltzmannMachines as RBMs

@testset "ReLU tail moments and hidden standardization (#276)" begin
    setprecision(BigFloat, 256) do
        for T in (Float32, Float64), a in T.((8, 77, 1000, 1e8)), γ in T.((0.25, 1, -4))
            θ = -a * sqrt(abs(γ))
            σ = inv(sqrt(abs(BigFloat(γ))))
            b = -BigFloat(θ) * σ
            λ = sqrt(2 / big(π)) / erfcx(b / sqrt(big(2)))
            μ_ref = T(σ * (λ - b))
            ν_ref = T(σ^2 * (1 - (λ - b) * λ))
            m2_ref = T(σ^2 * ((λ - b)^2 + 1 - (λ - b) * λ))
            layer = RBMs.ReLU(; θ = [θ], γ = [γ])
            rbm = RBMs.RBM(RBMs.Binary(T, (1,)), layer, zeros(T, 1, 1))
            for v in (zeros(T, 1), zeros(T, 1, 3), zeros(T, 1, 2, 3))
                μ, ν = @inferred RBMs.meanvar_from_inputs(layer, v)
                @test eltype(μ) == eltype(ν) == T
                @test all(isapprox.(μ, μ_ref; rtol = 16eps(T)))
                @test all(isapprox.(ν, ν_ref; rtol = 16eps(T)))
                moments = RBMs.moments_from_inputs(layer, v)
                @test all(isapprox.(selectdim(moments, 1, 2), m2_ref; rtol = 16eps(T)))
                @test RBMs.var_h_from_v(rbm, v) ≈ ν
                ∂ = RBMs.∂free_energy(rbm, v)
                @test ∂.hidden[1, 1] ≈ -μ_ref rtol = 16eps(T)
                @test ∂.hidden[2, 1] ≈ sign(γ) * m2_ref / 2 rtol = 16eps(T)

                std_rbm = RBMs.StandardizedRBM(deepcopy(rbm))
                @test RBMs.standardize_hidden_from_v!(std_rbm, v) === std_rbm
                @test only(std_rbm.offset_h) ≈ μ_ref rtol = 16eps(T)
                @test only(std_rbm.scale_h) ≈ sqrt(ν_ref) rtol = 16eps(T)
                @test all(isfinite, RBMs.free_energy(std_rbm, v))
            end

            # A symmetric dReLU combines two mirrored copies of the same tail.
            drelu = RBMs.dReLU(; θp = [θ], θn = [-θ], γp = [γ], γn = [γ])
            @test iszero(only(RBMs.mean_from_inputs(drelu)))
            @test only(RBMs.var_from_inputs(drelu)) ≈ m2_ref rtol = 16eps(T)
        end
    end
end
