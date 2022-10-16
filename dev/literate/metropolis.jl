import Makie
import CairoMakie
using Statistics: mean, std, var, cor
using Random: randn!, bitrand
using LogExpFunctions: logsumexp
using RestrictedBoltzmannMachines: BinaryRBM, energy, free_energy, metropolis!

N = 5
M = 2
rbm = BinaryRBM(randn(N), randn(M), randn(N, M) / √N);

β = 0.5
nsteps = 10000
nchains = 100
v = bitrand(N, nchains, nsteps)
metropolis!(v, rbm; β);

counts = Dict{BitVector, Int}()
for t in 1000:nsteps, n in 1:nchains
    counts[v[:,n,t]] = get(counts, v[:,n,t], 0) + 1
end
freqs = Dict(v => c / sum(values(counts)) for (v,c) in counts);

𝒱 = [BitVector(digits(Bool, x; base=2, pad=N)) for x in 0:(2^N - 1)];
ℱ = free_energy.(Ref(rbm), 𝒱);

# Commenting this until https://github.com/MakieOrg/Makie.jl/issues/2045 gets resolved.
# fig = Makie.Figure()
# ax = Makie.Axis(fig[1,1], xlabel="empirical freqs.", ylabel="log(p)", xscale=log10, yscale=log10)
# Makie.scatter!(ax, [get(freqs, v, 0.0) for v in 𝒱], exp.(-β * ℱ .- logsumexp(-β * ℱ)))
# Makie.scatter!(ax, [get(freqs, v, 0.0) for v in 𝒱], exp.(-ℱ .- logsumexp(-ℱ)))
# Makie.abline!(ax, 0, 1, color=:red)
# fig

fig = Makie.Figure()
ax = Makie.Axis(fig[1,1], xlabel="empirical freqs. (log)", ylabel="log(p)")
Makie.scatter!(ax, log.([get(freqs, v, 0.0) for v in 𝒱]), -β * ℱ .- logsumexp(-β * ℱ))
Makie.scatter!(ax, log.([get(freqs, v, 0.0) for v in 𝒱]), -ℱ .- logsumexp(-ℱ))
Makie.abline!(ax, 0, 1, color=:red)
fig

# Correlation

cor([get(freqs, v, 0.0) for v in 𝒱], exp.(-β * ℱ .- logsumexp(-β * ℱ)))
