# Reference implementations of the pseudolikelihood, computed from explicit
# substitution free-energy matrices. They are direct translations of the
# definition (flip each site and take free-energy differences), and serve as
# slow oracles to validate the optimized paths in `src/pseudolikelihood.jl`.
module PseudolikelihoodReference

using Adapt: adapt
using EllipsisNotation: (..)
using LogExpFunctions: logsumexp
using QuadGK: quadgk
using Statistics: mean
using RestrictedBoltzmannMachines: RBM, Binary, Spin, Potts, PottsGumbel, Gaussian,
    batch_size, colors, free_energy, sitedims, sitesize

_with_leading_dims(x::Number, n::Int) = x
_with_leading_dims(x::AbstractArray, n::Int) = reshape(x, ntuple(Returns(1), n)..., size(x)...)

"""
    log_pseudolikelihood_sites(rbm, v, sites)

Log-pseudolikelihood of a site conditioned on the other sites, where `sites`
is an array of site indices (CartesianIndex), one for each sample.
Returns an array of log-pseudolikelihood values, for each sample.
"""
function log_pseudolikelihood_sites(
        rbm::RBM, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
    )
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    ΔE = substitution_matrix_sites(rbm, v, sites)
    @assert size(ΔE) == (colors(rbm.visible), batch_size(rbm.visible, v)...)
    lPL = -logsumexp(-ΔE; dims = 1)
    @assert size(lPL) == (1, batch_size(rbm.visible, v)...)
    return reshape(lPL, batch_size(rbm.visible, v))
end

"""
    log_pseudolikelihood_exact(rbm, v)

Log-pseudolikelihood of `v`. This function computes the exact pseudolikelihood, doing
traces over all sites. Note that this can be slow for large number of samples.
"""
function log_pseudolikelihood_exact(rbm::RBM, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    ΔE = substitution_matrix_exhaustive(rbm, v)
    @assert size(ΔE) == (
        colors(rbm.visible), sitesize(rbm.visible)..., batch_size(rbm.visible, v)...,
    )
    lPLsites = -logsumexp(-ΔE; dims = 1)
    @assert size(lPLsites) == (1, sitesize(rbm.visible)..., batch_size(rbm.visible, v)...)
    lPL = mean(lPLsites; dims = 2:(sitedims(rbm.visible) + 1))
    return reshape(lPL, batch_size(rbm.visible, v))
end

"""
    substitution_matrix_sites(rbm, v, sites)

Returns an q x B matrix of free energies `F`, where `q` is the number of possible values
of each site, and `B` the number of data points. The entry `F[x,b]` equals the free energy
cost of flipping `site[b]` of `v[b]` to `x`, that is (schemetically):

    F[x, b] = free_energy(rbm, v_) - free_energy(rbm, v)

where `v = v[b]`, and `v_` is the same as `v` in all sites except `site[b]`,
where `v_` has the value `x`.
"""
function substitution_matrix_sites end

function _substitution_matrix_sites_2states(
        rbm::RBM, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}, states::Tuple
    )
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    F0 = free_energy(rbm, v)
    idx = [CartesianIndex(i, b) for (b, i) in pairs(sites)]
    E_ = similar(rbm.w, eltype(F0), (2, size(sites)...))
    for (k, x) in enumerate(states)
        v_ = copy(v)
        v_[idx] = fill(x, size(idx))
        E_[k, ..] .= free_energy(rbm, v_)
    end
    return E_ .- _with_leading_dims(F0, 1)
end

function substitution_matrix_sites(
        rbm::RBM{<:Binary}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
    )
    return _substitution_matrix_sites_2states(rbm, v, sites, (false, true))
end

function substitution_matrix_sites(
        rbm::RBM{<:Spin}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
    )
    return _substitution_matrix_sites_2states(rbm, v, sites, (Int8(-1), Int8(1)))
end

function substitution_matrix_sites(
        rbm::RBM{<:Potts}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
    )
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    q = colors(rbm.visible)
    F0 = free_energy(rbm, v)
    idx = [CartesianIndex(c, i, b) for c in 1:q, (b, i) in pairs(sites)]
    E_ = similar(rbm.w, eltype(F0), (q, size(sites)...))
    for x in 1:q
        v_ = copy(v)
        v_[idx] = [c == x for c in 1:q, _ in pairs(sites)]
        E_[x, ..] .= free_energy(rbm, v_)
    end
    return E_ .- _with_leading_dims(F0, 1)
end

function substitution_matrix_sites(
        rbm::RBM{<:PottsGumbel}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
    )
    return substitution_matrix_sites(RBM(Potts(rbm.visible), rbm.hidden, rbm.w), v, sites)
end

"""
    substitution_matrix_exhaustive(rbm, v)

Returns an q x N x B tensor of free energies `F`, where `q` is the number of possible
values of each site, `B` the number of data points, and `N` the sequence length:

````
q, N, B = size(v)
```

Thus `F` and `v` have the same size.
The entry `F[x,i,b]` gives the free energy cost of flipping site `i` to `x`
of `v[b]` from its original value to `x`, that is:

    F[x,i,b] = free_energy(rbm, v_) - free_energy(rbm, v[b])

where `v_` is the same as `v[b]` in all sites but `i`, where `v_` has the value `x`.

Note that `i` can be a set of indices.
"""
function substitution_matrix_exhaustive end

function _substitution_matrix_exhaustive_2states(rbm::RBM, v::AbstractArray, states::Tuple)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_indices = CartesianIndices(batch_size(rbm.visible, v))
    F0 = free_energy(rbm, v)
    E_ = similar(rbm.w, eltype(F0), (2, size(v)...))
    for i in CartesianIndices(size(rbm.visible))
        v_ = copy(v)
        for (k, x) in enumerate(states)
            v_[i, batch_indices] .= x
            E_[k, i, batch_indices] .= free_energy(rbm, v_)
        end
    end
    return E_ .- _with_leading_dims(F0, ndims(rbm.visible) + 1)
end

function substitution_matrix_exhaustive(rbm::RBM{<:Binary}, v::AbstractArray)
    return _substitution_matrix_exhaustive_2states(rbm, v, (false, true))
end

function substitution_matrix_exhaustive(rbm::RBM{<:Spin}, v::AbstractArray)
    return _substitution_matrix_exhaustive_2states(rbm, v, (Int8(-1), Int8(1)))
end

function substitution_matrix_exhaustive(rbm::RBM{<:Potts}, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_indices = CartesianIndices(batch_size(rbm.visible, v))
    F0 = free_energy(rbm, v)
    E_ = similar(rbm.w, eltype(F0), size(v))
    for i in CartesianIndices(sitesize(rbm.visible))
        v_ = copy(v)
        for x in 1:colors(rbm.visible)
            v_[:, i, batch_indices] .= false
            v_[x, i, batch_indices] .= true
            E_[x, i, batch_indices] .= free_energy(rbm, v_)
        end
    end
    return E_ .- _with_leading_dims(F0, sitedims(rbm.visible) + 1)
end

function substitution_matrix_exhaustive(rbm::RBM{<:PottsGumbel}, v::AbstractArray)
    return substitution_matrix_exhaustive(RBM(Potts(rbm.visible), rbm.hidden, rbm.w), v)
end

# Gaussian visible units: the conditional of a site is a density over the real line,
# obtained by numerical integration of the substituted free energies. Works with any
# hidden layer, so it serves as an oracle for the closed-form Gaussian-Gaussian path.
# The quadrature runs in Float64 whatever the model eltype, so that its tolerance is
# attainable and the oracle approximates the mathematical value.
function log_conditional_density_site(
        rbm::RBM{<:Gaussian}, v::AbstractArray, site::CartesianIndex
    )
    @assert size(v) == size(rbm.visible)
    rbm = adapt(Array{Float64}, rbm)
    v_ = Array{Float64}(v) # always a copy: the quadrature mutates it
    F0 = free_energy(rbm, v_)
    function F(x)
        v_[site] = x
        return free_energy(rbm, v_)
    end
    # Integrate over a wide finite domain, split into segments on the scale of the
    # visible unit's own Gaussian, so that the adaptive quadrature cannot miss a
    # narrow conditional peak. Infinite tails are avoided because the free energy
    # overflows to `Inf - Inf` at astronomically large inputs. The integrand is
    # shifted by its minimum on the grid to avoid overflow, and must vanish at the
    # ends of the domain for the truncation to be negligible.
    μ0 = rbm.visible.θ[site] / abs(rbm.visible.γ[site])
    σ0 = 1 / sqrt(abs(rbm.visible.γ[site]))
    grid = μ0 .+ σ0 .* range(-40, 40; step = 0.5)
    Fgrid = map(F, grid)
    Fmin = min(F0, minimum(Fgrid))
    @assert exp(-(first(Fgrid) - Fmin)) < 1.0e-12
    @assert exp(-(last(Fgrid) - Fmin)) < 1.0e-12
    integrand(x) = exp(-(F(x) - Fmin))
    Z, _ = quadgk(integrand, grid...; rtol = 1.0e-10)
    return Fmin - F0 - log(Z)
end

function log_pseudolikelihood_sites(
        rbm::RBM{<:Gaussian}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
    )
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    batch_indices = CartesianIndices(batch_size(rbm.visible, v))
    lPL = similar(rbm.w, float(eltype(rbm.w)), size(sites))
    for (b, site) in pairs(sites)
        vb = reshape(v[.., batch_indices[b]], size(rbm.visible))
        lPL[b] = log_conditional_density_site(rbm, vb, site)
    end
    return lPL
end

function log_pseudolikelihood_exact(rbm::RBM{<:Gaussian}, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_indices = CartesianIndices(batch_size(rbm.visible, v))
    lPL = similar(rbm.w, float(eltype(rbm.w)), batch_size(rbm.visible, v))
    for b in batch_indices
        vb = reshape(v[.., b], size(rbm.visible))
        lPL[b] = mean(
            log_conditional_density_site(rbm, vb, site)
                for site in CartesianIndices(size(rbm.visible))
        )
    end
    return lPL
end

end # module PseudolikelihoodReference
