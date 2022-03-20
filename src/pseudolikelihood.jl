"""
    log_pseudolikelihood(rbm, v; β = 1, exact = false)

Log-pseudolikelihood of `v`. If `exact` is `true`, the exact pseudolikelihood is returned.
But this is slow if `v` consists of many samples. Therefore by default `exact` is `false`,
in which case the result is a stochastic approximation, where a random site is selected
for each sample, and its conditional probability is calculated. In average the results
with `exact = false` coincide with the deterministic result, and the estimate is more
precise as the number of samples increases.
"""
function log_pseudolikelihood(rbm::AbstractRBM, v::AbstractArray; β::Real=true, exact::Bool=false)
    if exact
        return log_pseudolikelihood_exact(rbm, v; β)
    else
        return log_pseudolikelihood_stoch(rbm, v; β)
    end
end

"""
    log_pseudolikelihood_stoch(rbm, v; β = 1)

Log-pseudolikelihood of `v`. This function computes an stochastic approximation, by doing
a trace over random sites for each sample. For large number of samples, this is in average
close to the exact value of the pseudolikelihood.
"""
function log_pseudolikelihood_stoch(rbm::AbstractRBM, v::AbstractArray; β::Real=true)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    sites = [
        rand(CartesianIndices(sitesize(visible(rbm))))
        for _ in CartesianIndices(batch_size(visible(rbm), v))
    ]
    return log_pseudolikelihood_sites(rbm, v, sites; β)
end

"""
    log_pseudolikelihood_sites(rbm, v, sites; β = 1)

Log-pseudolikelihood of a site conditioned on the other sites, where `sites`
is an array of site indices (CartesianIndex), one for each sample.
Returns an array of log-pseudolikelihood values, for each sample.
"""
function log_pseudolikelihood_sites(
    rbm::AbstractRBM,
    v::AbstractArray,
    sites::AbstractArray{<:CartesianIndex};
    β::Real=true
)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    @assert size(sites) == batch_size(visible(rbm), v)
    ΔE = substitution_matrix_sites(rbm, v, sites; β)
    @assert size(ΔE) == (colors(visible(rbm)), batch_size(visible(rbm), v)...)
    lPL = -LogExpFunctions.logsumexp(-β * ΔE; dims=1)
    @assert size(lPL) == (1, batch_size(visible(rbm), v)...)
    return reshape(lPL, batch_size(visible(rbm), v))
end

"""
    log_pseudolikelihood_exact(rbm, v; β = 1)

Log-pseudolikelihood of `v`. This function computes the exact pseudolikelihood, doing
traces over all sites. Note that this can be slow for large number of samples.
"""
function log_pseudolikelihood_exact(rbm::AbstractRBM, v::AbstractArray; β::Real = true)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    ΔE = substitution_matrix_exhaustive(rbm, v; β)
    @assert size(ΔE) == (
        colors(visible(rbm)), sitesize(visible(rbm))..., batch_size(visible(rbm), v)...
    )
    lPLsites = -LogExpFunctions.logsumexp(-β * ΔE; dims=1)
    @assert size(lPLsites) == (1, sitesize(visible(rbm))..., batch_size(visible(rbm), v)...)
    lPL = Statistics.mean(lPLsites; dims=2:(sitedims(visible(rbm)) + 1))
    return reshape(lPL, batch_size(visible(rbm), v))
end

"""
    substitution_matrix_sites(rbm, v, sites; β = 1)

Returns an q x B matrix of free energies `F`, where `q` is the number of possible values
of each site, and `B` the number of data points. The entry `F[x,b]` equals the free energy
cost of flipping `site[b]` of `v[b]` to `x`, that is (schemetically):

    F[x, b] = free_energy(rbm, v_) - free_energy(rbm, v)

where `v = v[b]`, and `v_` is the same as `v` in all sites except `site[b]`,
where `v_` has the value `x`.
"""
function substitution_matrix_sites end

function substitution_matrix_sites(
    rbm::AbstractRBM{<:Binary},
    v::AbstractArray,
    sites::AbstractArray{<:CartesianIndex};
    β::Real = true
)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    @assert size(sites) == batch_size(visible(rbm), v)
    E_ = zeros(colors(visible(rbm)), batch_size(visible(rbm), v)...)
    for (k, x) in enumerate((false, true))
        v_ = copy(v)
        for (b, i) in pairs(sites)
            v_[i, b] = x
        end
        selectdim(E_, 1, k) .= free_energy(rbm, v_; β)
    end
    E = [E_[(v[i, b] > 0) + 1, b] for (b, i) in pairs(sites)]
    return E_ .- reshape(E, 1, batch_size(visible(rbm), v)...)
end

function substitution_matrix_sites(
    rbm::AbstractRBM{<:Spin},
    v::AbstractArray,
    sites::AbstractArray{<:CartesianIndex};
    β::Real = true
)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    @assert size(sites) == batch_size(visible(rbm), v)
    E_ = zeros(2, batch_size(visible(rbm), v)...)
    for (k, x) in enumerate((-1, 1))
        v_ = copy(v)
        for (b, i) in pairs(sites)
            v_[i, b] = x
        end
        selectdim(E_, 1, k) .= free_energy(rbm, v_; β)
    end
    E = [E_[(v[i, b] > 0) + 1, b] for (b, i) in pairs(sites)]
    return E_ .- reshape(E, 1, batch_size(visible(rbm), v)...)
end

function substitution_matrix_sites(
    rbm::AbstractRBM{<:Potts},
    v::AbstractArray,
    sites::AbstractArray{<:CartesianIndex};
    β::Real = true
)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    @assert size(sites) == batch_size(visible(rbm), v)
    E_ = zeros(colors(visible(rbm)), batch_size(visible(rbm), v)...)
    for x in 1:colors(visible(rbm))
        v_ = copy(v)
        for (b, i) in pairs(sites)
            v_[:, i, b] .= false
            v_[x, i, b] = true
        end
        selectdim(E_, 1, x) .= free_energy(rbm, v_; β)
    end
    c = onehot_decode(v)
    E = [E_[c[i, b], b] for (b, i) in pairs(sites)]
    return E_ .- reshape(E, 1, batch_size(visible(rbm), v)...)
end

"""
    substitution_matrix_exhaustive(rbm, v; β = 1)

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

function substitution_matrix_exhaustive(
    rbm::AbstractRBM{<:Binary}, v::AbstractArray; β::Real = true
)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    batch_indices = CartesianIndices(batch_size(visible(rbm), v))
    E_ = zeros(2, size(v)...)
    for i in CartesianIndices(size(visible(rbm)))
        v_ = copy(v)
        for (k, x) in enumerate((false, true))
            v_[i, batch_indices] .= x
            E_[k, i, batch_indices] .= free_energy(rbm, v_; β)
        end
    end
    E = [E_[(v[k] > 0) + 1, k] for k in CartesianIndices(v)]
    return E_ .- reshape(E, 1, size(v)...)
end

function substitution_matrix_exhaustive(
    rbm::AbstractRBM{<:Spin}, v::AbstractArray; β::Real = true
)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    batch_indices = CartesianIndices(batch_size(visible(rbm), v))
    E_ = zeros(2, size(v)...)
    for i in CartesianIndices(sitesize(visible(rbm)))
        v_ = copy(v)
        for (k, x) in enumerate((Int8(-1), Int8(1)))
            v_[i, batch_indices] .= x
            E_[k, i, batch_indices] .= free_energy(rbm, v_; β)
        end
    end
    E = [E_[(v[k] > 0) + 1, k] for k in CartesianIndices(v)]
    return E_ .- reshape(E, 1, size(v)...)
end

function substitution_matrix_exhaustive(rbm::AbstractRBM{<:Potts}, v::AbstractArray; β::Real = true)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    batch_indices = CartesianIndices(batch_size(visible(rbm), v))
    E_ = zeros(size(v))
    for i in CartesianIndices(sitesize(visible(rbm)))
        v_ = copy(v)
        for x in 1:colors(visible(rbm))
            v_[:, i, batch_indices] .= false
            v_[x, i, batch_indices] .= true
            E_[x, i, batch_indices] .= free_energy(rbm, v_; β)
        end
    end
    c = onehot_decode(v)
    E = [E_[c[k], k] for k in CartesianIndices(c)]
    return E_ .- reshape(E, 1, size(E)...)
end

#= ***
For Binary and Spin layers, a specialized log_pseudolikelihood_sites is a bit faster.
*** =#

function log_pseudolikelihood_sites(
    rbm::AbstractRBM{<:Binary},
    v::AbstractArray,
    sites::AbstractArray{<:CartesianIndex};
    β::Real = true
)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    @assert size(sites) == batch_size(visible(rbm), v)
    v_ = copy(v)
    for (b, i) in pairs(sites)
        v_[i, b] = 1 - v_[i, b]
    end
    F = free_energy(rbm, v; β)
    F_ = free_energy(rbm, v_; β)
    return -LogExpFunctions.log1pexp.(β * (F - F_))
end

function log_pseudolikelihood_sites(
    rbm::AbstractRBM{<:Spin},
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex};
    β::Real = true
)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    @assert size(sites) == batch_size(visible(rbm), v)
    v_ = copy(v)
    for (b, i) in pairs(sites)
        v_[i, b] = -v_[i, b]
    end
    F = free_energy(rbm, v; β)
    F_ = free_energy(rbm, v_; β)
    return -LogExpFunctions.log1pexp.(β * (F - F_))
end
