"""
    log_pseudolikelihood(rbm, v, β=1; exact=false)

Log-pseudolikelihood of `v`. If `exact` is `true`, the exact pseudolikelihood is returned.
But this is slow if `v` consists of many samples. Therefore by default `exact` is `false`,
in which case the result is a stochastic approximation, where a random site is selected
for each sample, and its conditional probability is calculated. In average the results
with `exact = false` coincide with the deterministic result, and the estimate is more
precise as the number of samples increases.
"""
function log_pseudolikelihood(rbm::RBM, v::AbstractArray, β::Real=true; exact::Bool=false)
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    if exact
        return log_pseudolikelihood_exact(rbm, v, β)
    else
        return log_pseudolikelihood_stoch(rbm, v, β)
    end
end

"""
    log_pseudolikelihood_stoch(rbm, v, β=1)

Log-pseudolikelihood of `v`. This function computes an stochastic approximation, by doing
a trace over random sites for each sample. For large number of samples, this is in average
close to the exact value of the pseudolikelihood.
"""
function log_pseudolikelihood_stoch(rbm::RBM, v::AbstractArray, β::Real = true)
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    all_sites = site_grid(rbm.visible)
    sites = [rand(all_sites) for _ in 1:size(v)[end]]
    return log_pseudolikelihood_sites(rbm, v, sites, β)
end

site_grid(layer::Potts) = CartesianIndices(size(layer)[2:end])
site_grid(layer) = CartesianIndices(size(layer))

"""
    log_pseudolikelihood_sites(rbm, v, sites, β=1)

Log-pseudolikelihood of a site conditioned on the other sites, where `sites`
is an array of site indices (CartesianIndex), one for each sample.
Returns an array of log-pseudolikelihood values, for each sample.
"""
function log_pseudolikelihood_sites(
    rbm::RBM,
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex},
    β::Real = true
)
    @assert size(v) == (size(rbm.visible)..., length(sites))
    ΔE = substitution_matrix_sites(rbm, v, sites, β)
    @assert size(ΔE) == (size(ΔE, 1), length(sites))
    lPL = -LogExpFunctions.logsumexp(-β * ΔE; dims=1)
    @assert size(lPL) == (1, length(sites))
    return vec(lPL)
end

"""
    log_pseudolikelihood_exact(rbm, v, β = 1)

Log-pseudolikelihood of `v`. This function computes the exact pseudolikelihood, doing
traces over all sites. Note that this can be slow for large number of samples.
"""
function log_pseudolikelihood_exact(rbm::RBM, v::AbstractArray, β::Real = true)
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    ΔE = substitution_matrix_exhaustive(rbm, v, β)
    @assert size(ΔE)[end] == size(v)[end]
    lPLsites = -LogExpFunctions.logsumexp(-β * ΔE; dims=1)
    lPL = mean(lPLsites; dims=ntuple(identity, ndims(lPLsites) - 1))
    return vec(lPL)
end

"""
    substitution_matrix_sites(rbm, v, sites, β = 1)

Returns an q x B matrix of free energies `F`, where `q` is the number of possible values
of each site, and `B` the number of data points. The entry `F[x,b]` equals the free energy
cost of flipping `site[b]` of `v[b]` to `x`, that is (schemetically):

    F[x, b] = free_energy(rbm, v_) - free_energy(rbm, v)

where `v = v[b]`, and `v_` is the same as `v` in all sites except `site[b]`,
where `v_` has the value `x`.
"""
function substitution_matrix_sites end

function substitution_matrix_sites(
    rbm::RBM{<:Binary},
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex},
    β::Real = true
)
    B = length(sites)
    @assert size(v) == (size(rbm.visible)..., B)
    E_ = zeros(2, B)
    for (k, x) in enumerate((false, true))
        v_ = copy(v)
        for (b, i) in enumerate(sites)
            v_[i, b] = x
        end
        E_[k,:] .= free_energy(rbm, v_, β)
    end
    E = [E_[(v[i, b] > 0) + 1, b] for (b, i) in enumerate(sites)]
    return E_ .- reshape(E, 1, B)
end

function substitution_matrix_sites(
    rbm::RBM{<:Spin},
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex},
    β::Real = true
)
    B = length(sites)
    @assert size(v) == (size(rbm.visible)..., B)
    E_ = zeros(2, B)
    for (k, x) in enumerate((-1, 1))
        v_ = copy(v)
        for (b, i) in enumerate(sites)
            v_[i, b] = x
        end
        E_[k,:] .= free_energy(rbm, v_, β)
    end
    E = [E_[(v[i, b] > 0) + 1, b] for (b, i) in enumerate(sites)]
    return E_ .- reshape(E, 1, B)
end

function substitution_matrix_sites(
    rbm::RBM{<:Potts},
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex},
    β::Real = true
)
    B = length(sites)
    @assert size(v) == (size(rbm.visible)..., B)
    E_ = zeros(rbm.visible.q, B)
    for x in 1:rbm.visible.q
        v_ = copy(v)
        for (b, i) in enumerate(sites)
            v_[:, i, b] .= false
            v_[x, i, b] = true
        end
        E_[x, :] .= free_energy(rbm, v_, β)
    end
    c = onehot_decode(v)
    E = [E_[c[i, b], b] for (b, i) in enumerate(sites)]
    return E_ .- reshape(E, 1, B)
end

"""
    substitution_matrix_exhaustive(rbm, v, β = 1)

Returns an q x N x B tensor of free energies `F`, where `q` is the number of possible
values of each site, `B` the number of data points, and `N` the sequence length:

````
q, N, B = size(v)
```

Thus `F` and `v` have the same size.
The entry `F[x,i,b]` gives the free energy cost of flipping site `i` to `x`
of `v[b]` from its original value to `x`, that is:

    F[x,i,b] = free_energy(rbm, v_, β) - free_energy(rbm, v[b], β)

where `v_` is the same as `v[b]` in all sites but `i`, where `v_` has the value `x`.

Note that `i` can be a set of indices.
"""
function substitution_matrix_exhaustive end

function substitution_matrix_exhaustive(
    rbm::RBM{<:Binary}, v::AbstractArray, β::Real = true
)
    B = size(v)[end]
    @assert size(v) == (size(rbm.visible)..., B)
    E_ = zeros(2, size(v)...)
    for i in site_grid(rbm.visible)
        v_ = copy(v)
        for (k, x) in enumerate((false, true))
            v_[i, :] .= x
            E_[k, i, :] .= free_energy(rbm, v_, β)
        end
    end
    E = [E_[(v[i,b] > 0) + 1, i, b] for i in site_grid(rbm.visible), b in 1:B]
    return E_ .- reshape(E, 1, size(v)...)
end

function substitution_matrix_exhaustive(
    rbm::RBM{<:Spin}, v::AbstractArray, β::Real = true
)
    B = size(v)[end]
    @assert size(v) == (size(rbm.visible)..., B)
    E_ = zeros(2, size(v)...)
    for i in site_grid(rbm.visible)
        v_ = copy(v)
        for (k, x) in enumerate((-1, 1))
            v_[i, :] .= x
            E_[k, i, :] .= free_energy(rbm, v_, β)
        end
    end
    E = [E_[(v[i,b] > 0) + 1, i, b] for i in site_grid(rbm.visible), b in 1:B]
    return E_ .- reshape(E, 1, size(v)...)
end

function substitution_matrix_exhaustive(
    rbm::RBM{<:Potts},
    v::AbstractArray,
    β::Real = true
)
    B = size(v)[end]
    @assert size(v) == (size(rbm.visible)..., B)
    E_ = zeros(size(v))
    for i in site_grid(rbm.visible)
        v_ = copy(v)
        for x in 1:rbm.visible.q
            v_[:, i, :] .= false
            v_[x, i, :] .= true
            E_[x, i, :] .= free_energy(rbm, v_, β)
        end
    end
    c = onehot_decode(v)
    E = [E_[c[i, b], i, b] for i in site_grid(rbm.visible), b in 1:B]
    return E_ .- reshape(E, 1, size(E)...)
end

#= ***
For Binary and Spin layers, a specialized log_pseudolikelihood_sites is a bit faster.
*** =#

function log_pseudolikelihood_sites(
    rbm::RBM{<:Binary},
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex},
    β::Real = true
)
    @assert size(v) == (size(rbm.visible)..., length(sites))
    v_ = copy(v)
    for (b, i) in enumerate(sites)
        v_[i, b] = 1 - v_[i, b]
    end
    F = free_energy(rbm, v, β)
    F_ = free_energy(rbm, v_, β)
    return -LogExpFunctions.log1pexp.(β * (F - F_))
end

function log_pseudolikelihood_sites(
    rbm::RBM{<:Spin},
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex},
    β::Real = true
)
    @assert size(v) == (size(rbm.visible)..., length(sites))
    v_ = copy(v)
    for (b, i) in enumerate(sites)
        v_[i, b] = -v_[i, b]
    end
    F = free_energy(rbm, v, β)
    F_ = free_energy(rbm, v_, β)
    return -LogExpFunctions.log1pexp.(β * (F - F_))
end
