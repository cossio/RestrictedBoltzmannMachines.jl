include("../tests_init.jl")

Random.seed!(18)

@inferred RBMs.randnt_half(1.0, 2.0)
@inferred RBMs.randnt_half(Float32(1.0), Float32(2.0))
@test RBMs.randnt_half(Float32(1.0), Float32(2.0)) isa Float32

# compare exact 1st and 2nd moments to Monte Carlo estimates
m1(μ,σ) = μ + σ * √(2/π) / SpecialFunctions.erfcx(-μ/σ/√2)
m2(μ,σ) = μ^2 + σ^2 + μ * σ * √(2/π) / SpecialFunctions.erfcx(-μ/σ/√2)

for μ = -1:1, σ = 1:2
    samples = [RBMs.randnt_half(μ,σ) for _ = 1:10^6]
    @test mean(samples.^1) ≈ m1(μ,σ) atol=1e-2
    @test mean(samples.^2) ≈ m2(μ,σ) atol=1e-2
end

# broadcasted versions
μ = 3randn(2,2); σ = 3rand(2,2)
dμ, dσ = Zygote.gradient(μ,σ) do μ,σ
    mean(m1.(μ,σ))
end
