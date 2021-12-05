# RestrictedBoltzmannMachines.jl Documentation

## Restricted Boltzmann Machines

We consider a restricted Boltzmann machine (RBM) with visible units $\mathbf{v} = (v_1, \ldots, v_N)$ and hidden units $\mathbf{h} = (h_1, \ldots, h_M)$.
The energy function is given by:

```math
E(\mathbf{v}, \mathbf{h}) = \sum_i \mathcal{V}_i(v_i) + \sum_{\mu}\mathcal{U}_{\mu}(h_{\mu}) - \sum_{i\mu} w_{i\mu} v_i h_{\mu}
```

where $\mathcal{V}_i(v_i)$ and $\mathcal{U}_{\mu}(h_{\mu})$ are the unit potentials and $w_{i\mu}$ the interaction weights.
The probability of a configuration is

```math
P(\mathbf{v}, \mathbf{h}) = \frac{1}{Z}\mathrm{e}^{-\beta E(\mathbf{v},\mathbf{h})}
```

where

```math
Z = \sum_{\mathbf{v}, \mathbf{h}} \mathrm{e}^{-\beta E(\mathbf{v}, \mathbf{h})}
```

is the partition function and $\beta$ the inverse temperature.
The machine assigns a likelihood:

```math
P(\mathbf{v}) = \underset{\mathbf{h}}{\sum} P (\mathbf{v}, \mathbf{h}) =
\frac{1}{Z} \mathrm{e}^{-\beta E_{\textrm{eff}}(\mathbf{v})}
```

to visible configurations, where $E_{\textrm{eff}}(\mathbf{v})$ is the free energy:

```math
E_{\textrm{eff}}(\mathbf{v}) = \sum_i \mathcal{V}_i(v_i) - \sum_{\mu}
\Gamma_{\mu} \left(\sum_i w_{i \mu} v_i \right)
```

and

```math
\Gamma_{\mu}(I) = \frac{1}{\beta} \ln \sum_{h_\mu} \mathrm{e}^{\beta(I h_{\mu} - \mathcal{U}_{\mu}(h_{\mu}))}
```

are the cumulant generating functions associated to the hidden unit potentials.

Note that $\beta$ refers to the inverse temperature in the distribution $P(\mathbf{v},\mathbf{h})$.
If instead we want to sample the marginal $P(\mathbf{v})$ at a different inverse temperature $\beta_v$,  we would have to use the distribution:

```math
P_{\beta_v}(\mathbf{v}) = \frac{\mathrm{e}^{- \beta_v E_\textrm{eff}
(\mathbf{v})}}{\sum_{\mathbf{v}} \mathrm{e}^{-\beta_{\mathrm{v}} E_{\textrm{eff}}
(\mathbf{v})}}
```

## Reference

This package doesn't export any symbols.
To make names shorter, we will import the package as `RBMs`:

```julia
import RestrictedBoltzmannMachines as RBMs
```

### RBMs

```@docs
RBMs.RBM
RBMs.energy
RBMs.free_energy
RBMs.interaction_energy
RBMs.inputs_v_to_h
RBMs.inputs_h_to_v
RBMs.sample_h_from_v
RBMs.sample_v_from_h
RBMs.sample_v_from_v
RBMs.sample_h_from_h
```

### Layers

```@docs
RBMs.Gaussian
```