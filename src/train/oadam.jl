#=
Optimistic ADAM. See Algorithm 1 in
https://par.nsf.gov/biblio/10079723-training-gans-optimism
=#

mutable struct OADAM
    eta::Float64
    beta::Tuple{Float64,Float64}
    state::IdDict
end
OADAM(η = 0.001, β = (0.9, 0.999)) = OADAM(η, β, IdDict())

function Flux.Optimise.apply!(o::OADAM, x, Δ)
    ϵ = Flux.Optimise.ϵ
    η, β = o.eta, o.beta
    mt, vt, βp, Δ_ = get!(o.state, x, (zero(x), zero(x), β, zero(x)))
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ = -Δ_
    @. Δ_ = η * mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ)
    @. Δ += 2Δ_
    o.state[x] = (mt, vt, βp .* β, Δ_)
    return Δ
end
