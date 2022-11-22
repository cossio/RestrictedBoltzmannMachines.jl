```@meta
EditURL = "<unknown>/src/literate/layers/ReLU.jl"
```

# ReLU layer

In this example we look at what the ReLU layer hidden units look like,
for different parameter values.

First load some packages.

````@example ReLU
import RestrictedBoltzmannMachines as RBMs
using CairoMakie, Statistics
nothing #hide
````

Now initialize our ReLU layer, with unit parameters spanning an interesting range.

````@example ReLU
θs = [0; 10]
γs = [5; 10]
layer = RBMs.ReLU(; θ = [θ for θ in θs, γ in γs], γ = [γ for θ in θs, γ in γs])
nothing #hide
````

Now we sample our layer to collect some data.

````@example ReLU
data = RBMs.sample_from_inputs(layer, zeros(size(layer)..., 10^6))
nothing #hide
````

Let's plot the resulting histogram of the activations of each unit.
We also overlay the analytical PDF.

````@example ReLU
fig = Figure(resolution=(700,500))
ax = Axis(fig[1,1])
xs = repeat(reshape(range(minimum(data), maximum(data), 100), 1, 1, 100), size(layer)...)
ps = exp.(-RBMs.cgfs(layer) .- RBMs.energies(layer, xs))
for (iθ, θ) in enumerate(θs), (iγ, γ) in enumerate(γs)
    hist!(ax, data[iθ, iγ, :], normalization=:pdf, label="θ=$θ, γ=$γ")
    lines!(xs[iθ, iγ, :], ps[iθ, iγ, :], linewidth=2)
end
axislegend(ax)
fig
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

