#=
Effect of decaying the learning rate during training to achieve convergence.
=#

using CairoMakie, Statistics, Random, LinearAlgebra
import Flux, MLDatasets, ValueHistories
import RestrictedBoltzmannMachines as RBMs

#=
Load MNIST dataset.
=#

Float = Float32;
train_x, train_y = MLDatasets.MNIST.traindata();
tests_x, tests_y = MLDatasets.MNIST.testdata();
train_x = Float.(train_x .≥ 0.5);
tests_x = Float.(tests_x .≥ 0.5);
# select only 0s digits
train_x = train_x[:,:, train_y .== 0]
tests_x = tests_x[:,:, tests_y .== 0]
train_y = train_y[train_y .== 0]
tests_y = tests_y[tests_y .== 0]
train_nsamples = length(train_y)
tests_nsamples = length(tests_y)
nothing #hide

#=
Consider first an RBM that we train without decaying the learning rate.
We will train this machine for 300 epochs.
=#

rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,100), zeros(Float,28,28,100));
RBMs.initialize!(rbm, train_x);
opt = Flux.ADAM(0.001f0, (0.9f0, 0.999f0))
history = ValueHistories.MVHistory()
RBMs.pcd!(rbm, train_x; history=history,
    epochs=300, batchsize=128, verbose=true, steps=1,
    optimizer=opt
)
nothing #hide

#=
Now train an RBM with 200 normal epochs, followed by 100 epochs where the
learning-rate is cut in half every 10 epochs.
=#

rbm_decay = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,100), zeros(Float,28,28,100));
RBMs.initialize!(rbm_decay, train_x);
opt = Flux.ADAM(0.001f0, (0.9f0, 0.999f0))
history_decay = ValueHistories.MVHistory()
RBMs.pcd!(rbm_decay, train_x; history=history_decay,
    epochs=200, batchsize=128, verbose=true, steps=1,
    optimizer=opt
)
for meta_epoch = 1:10
    opt.eta /= 2
    RBMs.pcd!(rbm_decay, train_x; history=history_decay,
        epochs=10, batchsize=128, verbose=true, steps=1,
        optimizer=opt
    )
end
nothing #hide

#=
Compare the results
=#

fig = Figure(resolution=(600,400))
ax = Axis(fig[1,1])
lines!(ax, get(history, :lpl)..., label="normal")
lines!(ax, get(history_decay, :lpl)..., label="decay")
axislegend(ax, position=:rb)
fig

#=
Check convergence by computing the moment-matching conditions.
First generate MC data from the RBMs.
=#

samples_v = RBMs.sample_v_from_v(rbm, tests_x; steps=1000)
samples_h = RBMs.sample_h_from_v(rbm, samples_v)
samples_v_decay = RBMs.sample_v_from_v(rbm_decay, tests_x; steps=1000)
samples_h_decay = RBMs.sample_h_from_v(rbm_decay, samples_v_decay)
nothing #hide

#=
Now make the plots
=#

#=
For the RBM with constant learning-rate.
=#

fig = Figure(resolution=(900, 300))
ax = Axis(fig[1,1])
heatmap!(ax, mean(train_x, dims=3)[:,:,1])
hidedecorations!(ax)
ax = Axis(fig[1,2])
heatmap!(ax, mean(tests_x, dims=3)[:,:,1])
hidedecorations!(ax)
ax = Axis(fig[1,3])
heatmap!(ax, mean(samples_v, dims=3)[:,:,1])
hidedecorations!(ax)
fig

#=
Moment matching conditions
=#

h_data = RBMs.mean_h_from_v(rbm, train_x);
h_model = RBMs.mean_h_from_v(rbm, samples_v);

fig = Figure(resolution=(900, 300))

ax = Axis(fig[1,1], xlabel=L"\langle v_i \rangle_\mathrm{data}", ylabel=L"\langle v_i \rangle_\mathrm{model}", limits=(0,1,0,1))
scatter!(ax, vec(mean(train_x; dims=3)), vec(mean(samples_v; dims=3)))
abline!(ax, 0, 1; color=:red)

ax = Axis(fig[1,2], xlabel=L"\langle h_\mu \rangle_\mathrm{data}", ylabel=L"\langle h_\mu \rangle_\mathrm{model}", limits=(0,1,0,1))
scatter!(ax, vec(mean(h_data; dims=2)), vec(mean(h_model; dims=2)))
abline!(ax, 0, 1; color=:red)

ax = Axis(fig[1,3], xlabel=L"\langle v_i h_\mu \rangle_\mathrm{data}", ylabel=L"\langle v_i h_\mu \rangle_\mathrm{model}", limits=(0,1,0,1))
scatter!(ax,
    vec([dot(train_x[i,j,:], h_data[μ,:]) / size(train_x,3) for i=1:28, j=1:28, μ=1:100]),
    vec([dot(samples_v[i,j,:], h_model[μ,:]) / size(samples_v,3) for i=1:28, j=1:28, μ=1:100])
)
abline!(ax, 0, 1; color=:red)

fig

#=
For the RBM with decaying learning-rate.
=#

fig = Figure(resolution=(900, 300))
ax = Axis(fig[1,1])
heatmap!(ax, mean(train_x, dims=3)[:,:,1])
hidedecorations!(ax)
ax = Axis(fig[1,2])
heatmap!(ax, mean(tests_x, dims=3)[:,:,1])
hidedecorations!(ax)
ax = Axis(fig[1,3])
heatmap!(ax, mean(samples_v_decay, dims=3)[:,:,1])
hidedecorations!(ax)
fig

#=
Moment matching conditions
=#

h_data = RBMs.mean_h_from_v(rbm, train_x);
h_model = RBMs.mean_h_from_v(rbm, samples_v_decay);

fig = Figure(resolution=(900, 300))

ax = Axis(fig[1,1], xlabel=L"\langle v_i \rangle_\mathrm{data}", ylabel=L"\langle v_i \rangle_\mathrm{model}", limits=(0,1,0,1))
scatter!(ax, vec(mean(train_x; dims=3)), vec(mean(samples_v_decay; dims=3)))
abline!(ax, 0, 1; color=:red)

ax = Axis(fig[1,2], xlabel=L"\langle h_\mu \rangle_\mathrm{data}", ylabel=L"\langle h_\mu \rangle_\mathrm{model}", limits=(0,1,0,1))
scatter!(ax, vec(mean(h_data; dims=2)), vec(mean(h_model; dims=2)))
abline!(ax, 0, 1; color=:red)

ax = Axis(fig[1,3], xlabel=L"\langle v_i h_\mu \rangle_\mathrm{data}", ylabel=L"\langle v_i h_\mu \rangle_\mathrm{model}", limits=(0,1,0,1))
scatter!(ax,
    vec([dot(train_x[i,j,:], h_data[μ,:]) / size(train_x,3) for i=1:28, j=1:28, μ=1:100]),
    vec([dot(samples_v_decay[i,j,:], h_model[μ,:]) / size(samples_v_decay,3) for i=1:28, j=1:28, μ=1:100])
)
abline!(ax, 0, 1; color=:red)

fig
