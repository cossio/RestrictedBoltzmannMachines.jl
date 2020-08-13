using Test, Random, Statistics, Zygote, SpecialFunctions,
    RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: randnt, randnt_half, ∇randnt_half

Random.seed!(18)

@inferred randnt_half(1.0, 2.0)
@inferred randnt_half(Float32(1.0), Float32(2.0))
@test randnt_half(Float32(1.0), Float32(2.0)) isa Float32

# compare exact 1st and 2nd moments to Monte Carlo estimates
m1(μ,σ) = μ + σ * √(2/π) / erfcx(-μ/σ/√2)
m2(μ,σ) = μ^2 + σ^2 + μ * σ * √(2/π) / erfcx(-μ/σ/√2)

for μ = -1:1, σ = 1:2
    samples = [randnt_half(μ,σ) for _ = 1:10^6]
    @test mean(samples.^1) ≈ m1(μ,σ) atol=1e-2
    @test mean(samples.^2) ≈ m2(μ,σ) atol=1e-2

    dμ, dσ = gradient(m1, μ, σ)
    dμmc, dσmc = gradient(μ,σ) do μ,σ
        mean([randnt_half(μ,σ) for _ = 1:10^4])
    end
    @test dμmc ≈ dμ atol=0.1
    @test dσmc ≈ dσ atol=0.1

    dμ, dσ = gradient(μ,σ) do μ,σ
        sqrt(m2(μ,σ) - m1(μ,σ)^2)
    end
    dμmc, dσmc = gradient(μ,σ) do μ,σ
        samples = [randnt_half(μ,σ) for _ = 1:10^4]
        std(samples)
    end
    @test dμmc ≈ dμ atol=0.1
    @test dσmc ≈ dσ atol=0.1
end

# broadcasted versions
μ = 3randn(2,2); σ = 3rand(2,2)
dμ, dσ = gradient(μ,σ) do μ,σ
    mean(m1.(μ,σ))
end
dμmc, dσmc = gradient(μ,σ) do μ,σ
    m = zero(randnt_half.(μ,σ))
    for _ = 1:10^4
        m += randnt_half.(μ,σ)
    end
    mean(m / 10^4)
end
@test dμmc ≈ dμ atol=0.1
@test dσmc ≈ dσ atol=0.1

μ = 3randn(2,2); σ = 3rand(2,2)
dμ, dσ = gradient(μ,σ) do μ,σ
    mean(@. m2(μ,σ))
end
dμmc, dσmc = gradient(μ,σ) do μ,σ
    m = zero(randnt_half.(μ,σ))
    for _ = 1:10^4
        m += @. randnt_half(μ,σ)^2
    end
    mean(@. m / 10^4)
end
@test dμmc ≈ dμ atol=0.1
@test dσmc ≈ dσ atol=0.1
