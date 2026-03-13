"""
    log_pseudolikelihood(rbm, v; exact = false)

Log-pseudolikelihood of `v`. If `exact` is `true`, the exact pseudolikelihood is returned.
But this is slow if `v` consists of many samples. Therefore by default `exact` is `false`,
in which case the result is a stochastic approximation, where a random site is selected
for each sample, and its conditional probability is calculated. In average the results
with `exact = false` coincide with the deterministic result, and the estimate is more
precise as the number of samples increases.
"""
function log_pseudolikelihood(rbm::RBM, v::AbstractArray; exact::Bool=false)
    if exact
        return log_pseudolikelihood_exact(rbm, v)
    else
        return log_pseudolikelihood_stoch(rbm, v)
    end
end

"""
    log_pseudolikelihood_stoch(rbm, v)

Log-pseudolikelihood of `v`. This function computes an stochastic approximation, by doing
a trace over random sites for each sample. For large number of samples, this is in average
close to the exact value of the pseudolikelihood.
"""
function log_pseudolikelihood_stoch(rbm::RBM, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_sz = batch_size(rbm.visible, v)
    sites = reshape([
        rand(CartesianIndices(sitesize(rbm.visible)))
        for _ in 1:prod(batch_sz)
    ], batch_sz)
    return log_pseudolikelihood_sites(rbm, v, sites)
end

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
    lPL = -logsumexp(-ΔE; dims=1)
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
        colors(rbm.visible), sitesize(rbm.visible)..., batch_size(rbm.visible, v)...
    )
    lPLsites = -logsumexp(-ΔE; dims=1)
    @assert size(lPLsites) == (1, sitesize(rbm.visible)..., batch_size(rbm.visible, v)...)
    lPL = mean(lPLsites; dims=2:(sitedims(rbm.visible) + 1))
    return reshape(lPL, batch_size(rbm.visible, v))
end

function _pseudolikelihood_context(rbm::RBM, v::AbstractArray)
    batch_sz = batch_size(rbm.visible, v)
    B = prod(batch_sz)
    inputs = inputs_h_from_v(rbm, v)
    Iflat = reshape(flatten(rbm.hidden, inputs), length(rbm.hidden), B)
    Γ = cgf(rbm.hidden, inputs)
    Γ0 = isempty(batch_sz) ? fill(Γ, 1) : vec(Γ)
    return batch_sz, B, Iflat, Γ0
end

@inline _potts_site_index(x::Integer, i::CartesianIndex) = (x, Tuple(i)...)

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

function substitution_matrix_sites(
    rbm::RBM{<:Binary}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    E_ = zeros(2, batch_size(rbm.visible, v)...)
    for (k, x) in enumerate((false, true))
        v_ = copy(v)
        for (b, i) in pairs(sites)
            v_[i, b] = x
        end
        E_[k, ..] .= free_energy(rbm, v_)
    end
    E = [E_[(v[i, b] > 0) + 1, b] for (b, i) in pairs(sites)]
    return E_ .- reshape(E, 1, batch_size(rbm.visible, v)...)
end

function substitution_matrix_sites(
    rbm::RBM{<:Spin}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    E_ = zeros(2, batch_size(rbm.visible, v)...)
    for (k, x) in enumerate((-1, 1))
        v_ = copy(v)
        for (b, i) in pairs(sites)
            v_[i, b] = x
        end
        E_[k, ..] .= free_energy(rbm, v_)
    end
    E = [E_[(v[i, b] > 0) + 1, b] for (b, i) in pairs(sites)]
    return E_ .- reshape(E, 1, batch_size(rbm.visible, v)...)
end

function substitution_matrix_sites(
    rbm::RBM{<:Potts}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    E_ = zeros(size(rbm.visible, 1), batch_size(rbm.visible, v)...)
    for k in 1:size(rbm.visible, 1)
        v_ = copy(v)
        for (b, i) in pairs(sites)
            v_[:, i, b] .= false
            v_[k, i, b] = true
        end
        E_[k, ..] .= free_energy(rbm, v_)
    end
    c = onehot_decode(v)
    E = [E_[c[i, b], b] for (b, i) in pairs(sites)]
    return E_ .- reshape(E, 1, batch_size(rbm.visible, v)...)
end

function substitution_matrix_sites(rbm::RBM{<:PottsGumbel}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex})
    substitution_matrix_sites(RBM(Potts(rbm.visible), rbm.hidden, rbm.w), v, sites)
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

function substitution_matrix_exhaustive(rbm::RBM{<:Binary}, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_indices = CartesianIndices(batch_size(rbm.visible, v))
    E_ = zeros(2, size(v)...)
    for i in CartesianIndices(size(rbm.visible))
        v_ = copy(v)
        for (k, x) in enumerate((false, true))
            v_[i, batch_indices] .= x
            E_[k, i, batch_indices] .= free_energy(rbm, v_)
        end
    end
    E = [E_[(v[k] > 0) + 1, k] for k in CartesianIndices(v)]
    return E_ .- reshape(E, 1, size(v)...)
end

function substitution_matrix_exhaustive(rbm::RBM{<:Spin}, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_indices = CartesianIndices(batch_size(rbm.visible, v))
    E_ = zeros(2, size(v)...)
    for i in CartesianIndices(sitesize(rbm.visible))
        v_ = copy(v)
        for (k, x) in enumerate((Int8(-1), Int8(1)))
            v_[i, batch_indices] .= x
            E_[k, i, batch_indices] .= free_energy(rbm, v_)
        end
    end
    E = [E_[(v[k] > 0) + 1, k] for k in CartesianIndices(v)]
    return E_ .- reshape(E, 1, size(v)...)
end

function substitution_matrix_exhaustive(rbm::RBM{<:Potts}, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_indices = CartesianIndices(batch_size(rbm.visible, v))
    E_ = zeros(size(v))
    for i in CartesianIndices(sitesize(rbm.visible))
        v_ = copy(v)
        for x in 1:size(rbm.visible, 1)
            v_[:, i, batch_indices] .= false
            v_[x, i, batch_indices] .= true
            E_[x, i, batch_indices] .= free_energy(rbm, v_)
        end
    end
    c = onehot_decode(v)
    E = [E_[c[k], k] for k in CartesianIndices(c)]
    return E_ .- reshape(E, 1, size(E)...)
end

function substitution_matrix_exhaustive(rbm::RBM{<:PottsGumbel}, v::AbstractArray)
    substitution_matrix_exhaustive(RBM(Potts(rbm.visible), rbm.hidden, rbm.w), v)
end

function log_pseudolikelihood_sites(
    rbm::RBM{<:Binary}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    batch_sz, B, Iflat, Γ0 = _pseudolikelihood_context(rbm, v)
    vbatch = reshape(v, size(rbm.visible)..., B)
    sitesvec = reshape(sites, B)
    site_linear = LinearIndices(size(rbm.visible))
    buffer = similar(Iflat, size(Iflat, 1))
    out = similar(Γ0, B)
    buffer_hidden = reshape(buffer, size(rbm.hidden))
    wflat = flat_w(rbm)

    @views for b in 1:B
        i = sitesvec[b]
        j = site_linear[i]
        δ = ifelse(vbatch[i, b] > 0, -1, 1)
        wrow = view(wflat, j, :)
        icol = view(Iflat, :, b)
        @. buffer = icol + δ * wrow
        ΔΓ = cgf(rbm.hidden, buffer_hidden) - Γ0[b]
        ΔF = -rbm.visible.θ[i] * δ - ΔΓ
        out[b] = -log1pexp(-ΔF)
    end
    return reshape(out, batch_sz)
end

function log_pseudolikelihood_sites(
    rbm::RBM{<:Spin}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    batch_sz, B, Iflat, Γ0 = _pseudolikelihood_context(rbm, v)
    vbatch = reshape(v, size(rbm.visible)..., B)
    sitesvec = reshape(sites, B)
    site_linear = LinearIndices(size(rbm.visible))
    buffer = similar(Iflat, size(Iflat, 1))
    out = similar(Γ0, B)
    buffer_hidden = reshape(buffer, size(rbm.hidden))
    wflat = flat_w(rbm)

    @views for b in 1:B
        i = sitesvec[b]
        j = site_linear[i]
        δ = ifelse(vbatch[i, b] > 0, -2, 2)
        wrow = view(wflat, j, :)
        icol = view(Iflat, :, b)
        @. buffer = icol + δ * wrow
        ΔΓ = cgf(rbm.hidden, buffer_hidden) - Γ0[b]
        ΔF = -rbm.visible.θ[i] * δ - ΔΓ
        out[b] = -log1pexp(-ΔF)
    end
    return reshape(out, batch_sz)
end

function log_pseudolikelihood_sites(
    rbm::RBM{<:Potts}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    batch_sz, B, Iflat, Γ0 = _pseudolikelihood_context(rbm, v)
    q = colors(rbm.visible)
    cats = reshape(onehot_decode(v), sitesize(rbm.visible)..., B)
    sitesvec = reshape(sites, B)
    visible_linear = LinearIndices(size(rbm.visible))
    buffer = similar(Iflat, size(Iflat, 1))
    ΔF = similar(Γ0, q)
    out = similar(Γ0, B)
    buffer_hidden = reshape(buffer, size(rbm.hidden))
    wflat = flat_w(rbm)

    @views for b in 1:B
        i = sitesvec[b]
        old = cats[i, b]
        old_idx = _potts_site_index(old, i)
        old_row = visible_linear[old_idx...]
        old_w = view(wflat, old_row, :)
        old_field = rbm.visible.θ[old_idx...]
        icol = view(Iflat, :, b)
        for x in 1:q
            if x == old
                ΔF[x] = zero(eltype(ΔF))
            else
                x_idx = _potts_site_index(x, i)
                x_row = visible_linear[x_idx...]
                x_w = view(wflat, x_row, :)
                @. buffer = icol + x_w - old_w
                ΔΓ = cgf(rbm.hidden, buffer_hidden) - Γ0[b]
                ΔF[x] = old_field - rbm.visible.θ[x_idx...] - ΔΓ
            end
        end
        out[b] = -logsumexp(-ΔF)
    end

    return reshape(out, batch_sz)
end

function log_pseudolikelihood_sites(
    rbm::RBM{<:PottsGumbel}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
)
    return log_pseudolikelihood_sites(RBM(Potts(rbm.visible), rbm.hidden, rbm.w), v, sites)
end

function log_pseudolikelihood_exact(rbm::RBM{<:Binary}, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_sz, B, Iflat, Γ0 = _pseudolikelihood_context(rbm, v)
    vbatch = reshape(v, size(rbm.visible)..., B)
    site_linear = LinearIndices(size(rbm.visible))
    buffer = similar(Iflat, size(Iflat, 1))
    lPL = zeros(eltype(Γ0), B)
    buffer_hidden = reshape(buffer, size(rbm.hidden))
    wflat = flat_w(rbm)

    @views for i in CartesianIndices(size(rbm.visible))
        j = site_linear[i]
        θi = rbm.visible.θ[i]
        wrow = view(wflat, j, :)
        for b in 1:B
            δ = ifelse(vbatch[i, b] > 0, -1, 1)
            icol = view(Iflat, :, b)
            @. buffer = icol + δ * wrow
            ΔΓ = cgf(rbm.hidden, buffer_hidden) - Γ0[b]
            ΔF = -θi * δ - ΔΓ
            lPL[b] += -log1pexp(-ΔF)
        end
    end

    lPL ./= prod(sitesize(rbm.visible))
    return reshape(lPL, batch_sz)
end

function log_pseudolikelihood_exact(rbm::RBM{<:Spin}, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_sz, B, Iflat, Γ0 = _pseudolikelihood_context(rbm, v)
    vbatch = reshape(v, size(rbm.visible)..., B)
    site_linear = LinearIndices(size(rbm.visible))
    buffer = similar(Iflat, size(Iflat, 1))
    lPL = zeros(eltype(Γ0), B)
    buffer_hidden = reshape(buffer, size(rbm.hidden))
    wflat = flat_w(rbm)

    @views for i in CartesianIndices(size(rbm.visible))
        j = site_linear[i]
        θi = rbm.visible.θ[i]
        wrow = view(wflat, j, :)
        for b in 1:B
            δ = ifelse(vbatch[i, b] > 0, -2, 2)
            icol = view(Iflat, :, b)
            @. buffer = icol + δ * wrow
            ΔΓ = cgf(rbm.hidden, buffer_hidden) - Γ0[b]
            ΔF = -θi * δ - ΔΓ
            lPL[b] += -log1pexp(-ΔF)
        end
    end

    lPL ./= prod(sitesize(rbm.visible))
    return reshape(lPL, batch_sz)
end

function log_pseudolikelihood_exact(rbm::RBM{<:Potts}, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_sz, B, Iflat, Γ0 = _pseudolikelihood_context(rbm, v)
    q = colors(rbm.visible)
    cats = reshape(onehot_decode(v), sitesize(rbm.visible)..., B)
    visible_linear = LinearIndices(size(rbm.visible))
    buffer = similar(Iflat, size(Iflat, 1))
    ΔF = similar(Γ0, q)
    lPL = zeros(eltype(Γ0), B)
    buffer_hidden = reshape(buffer, size(rbm.hidden))
    wflat = flat_w(rbm)

    @views for i in CartesianIndices(sitesize(rbm.visible))
        for b in 1:B
            old = cats[i, b]
            old_idx = _potts_site_index(old, i)
            old_row = visible_linear[old_idx...]
            old_w = view(wflat, old_row, :)
            old_field = rbm.visible.θ[old_idx...]
            icol = view(Iflat, :, b)
            for x in 1:q
                if x == old
                    ΔF[x] = zero(eltype(ΔF))
                else
                    x_idx = _potts_site_index(x, i)
                    x_row = visible_linear[x_idx...]
                    x_w = view(wflat, x_row, :)
                    @. buffer = icol + x_w - old_w
                    ΔΓ = cgf(rbm.hidden, buffer_hidden) - Γ0[b]
                    ΔF[x] = old_field - rbm.visible.θ[x_idx...] - ΔΓ
                end
            end
            lPL[b] += -logsumexp(-ΔF)
        end
    end

    lPL ./= prod(sitesize(rbm.visible))
    return reshape(lPL, batch_sz)
end

function log_pseudolikelihood_exact(rbm::RBM{<:PottsGumbel}, v::AbstractArray)
    return log_pseudolikelihood_exact(RBM(Potts(rbm.visible), rbm.hidden, rbm.w), v)
end
