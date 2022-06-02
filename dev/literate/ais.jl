#=
# Annealed importance sampling

We can compute the partition function of the RBM (and hence the log-likelihood) with
annealed importance sampling (AIS).
=#

import MLDatasets
import Makie
import CairoMakie
import RestrictedBoltzmannMachines as RBMs
using Statistics: mean, std, middle
using ValueHistories: MVHistory
using RestrictedBoltzmannMachines: Binary, BinaryRBM, initialize!, pcd!,
    aise, raise, logmeanexp, logstdexp, sample_v_from_v

# Load MNIST (0 digit only).

Float = Float32
train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float}(train_x[:, :, train_y .== 0] .> 0.5)
nothing #hide

# Train an RBM

rbm = BinaryRBM(Float, (28,28), 128)
initialize!(rbm, train_x)
@time pcd!(rbm, train_x; epochs=100, batchsize=128)
nothing #hide

# Estimate Z with AIS and reverse AIS.

nsamples=100
ndists = [10, 100, 1000, 10000, 100000]
R_ais = Vector{Float64}[]
R_rev = Vector{Float64}[]
for nbetas in ndists
    push!(R_ais,
        @time aise(rbm; nbetas, nsamples, init=initialize!(Binary(zero(rbm.visible.θ)), train_x))
    )
    v = train_x[:, :, rand(1:size(train_x, 3), nsamples)]
    sample_v_from_v(rbm, v; steps=1000) # equilibrate
    push!(R_rev,
        @time raise(rbm; v, nbetas, init=initialize!(Binary(zero(rbm.visible.θ)), train_x))
    )
end

# Plots

fig = Makie.Figure()
ax = Makie.Axis(
    fig[1,1], width=700, height=400, xscale=log10, xlabel="interpolating distributions", ylabel="log(Z)"
)
Makie.band!(
    ax, ndists,
    mean.(R_ais) - std.(R_ais),
    mean.(R_ais) + std.(R_ais);
    color=(:blue, 0.25)
)
Makie.band!(
    ax, ndists,
    mean.(R_rev) - std.(R_rev),
    mean.(R_rev) + std.(R_rev);
    color=(:black, 0.25)
)
Makie.lines!(ax, ndists, mean.(R_ais); color=:blue, label="AIS")
Makie.lines!(ax, ndists, mean.(R_rev); color=:black, label="reverse AIS")
Makie.lines!(ax, ndists, logmeanexp.(R_ais); color=:blue, linestyle=:dash)
Makie.lines!(ax, ndists, logmeanexp.(R_rev); color=:black, linestyle=:dash)
Makie.hlines!(ax, middle(mean(R_ais[end]), mean(R_rev[end])), linestyle=:dash, color=:red, label="limiting estimate")
Makie.xlims!(extrema(ndists)...)
Makie.axislegend(ax, position=:rb)
Makie.resize_to_layout!(fig)
fig
