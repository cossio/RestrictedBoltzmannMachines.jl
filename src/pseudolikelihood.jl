"""
    log_pseudolikelihood(rbm, v; exact = false)

Log-pseudolikelihood of `v`. If `exact` is `true`, the exact pseudolikelihood is returned.
But this is slow if `v` consists of many samples. Therefore by default `exact` is `false`,
in which case the result is a stochastic approximation, where a random site is selected
for each sample, and its conditional probability is calculated. In average the results
with `exact = false` coincide with the deterministic result, and the estimate is more
precise as the number of samples increases.
"""
function log_pseudolikelihood(rbm::RBM, v::AbstractArray; exact::Bool = false)
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
    sites = reshape(
        [
            rand(CartesianIndices(sitesize(rbm.visible)))
                for _ in 1:prod(batch_sz)
        ], batch_sz
    )
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

# Copy an array of indices (or values) to the same kind of array as `template`
# (e.g. a GPU array), so that gather / scatter operations don't mix host and
# device arrays. On CPU this is a plain copy; the arrays passed here are small
# (O(batch) index lists), so the overhead is negligible.
_on_device(template::AbstractArray, x::AbstractArray) =
    copyto!(similar(template, eltype(x), size(x)), x)

_with_leading_dims(x::Number, n::Int) = x
_with_leading_dims(x::AbstractArray, n::Int) = reshape(x, ntuple(Returns(1), n)..., size(x)...)

function _pseudolikelihood_context(rbm::RBM, v::AbstractArray)
    batch_sz = batch_size(rbm.visible, v)
    B = prod(batch_sz)
    vB = reshape(v, size(rbm.visible)..., B)
    inputs = inputs_h_from_v(rbm, vB)
    Iflat = reshape(inputs, length(rbm.hidden), B)
    Γ0 = vec(cgf(rbm.hidden, inputs))
    return batch_sz, B, vB, Iflat, Γ0
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
    idx = _on_device(rbm.w, [CartesianIndex(i, b) for (b, i) in pairs(sites)])
    E_ = similar(rbm.w, eltype(F0), (2, size(sites)...))
    for (k, x) in enumerate(states)
        v_ = copy(v)
        v_[idx] = _on_device(v, fill(x, size(idx)))
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
    idx = _on_device(rbm.w, [CartesianIndex(c, i, b) for c in 1:q, (b, i) in pairs(sites)])
    E_ = similar(rbm.w, eltype(F0), (q, size(sites)...))
    for x in 1:q
        v_ = copy(v)
        v_[idx] = _on_device(v, [c == x for c in 1:q, _ in pairs(sites)])
        E_[x, ..] .= free_energy(rbm, v_)
    end
    return E_ .- _with_leading_dims(F0, 1)
end

function substitution_matrix_sites(rbm::RBM{<:PottsGumbel}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex})
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

# Binary and Spin units differ only in the input shift produced by a flip:
# 0 -> 1 changes the input to hidden units by +w (flip = 1),
# -1 -> 1 changes it by +2w (flip = 2).
function _log_pseudolikelihood_sites_2states(
        rbm::RBM, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}, flip::Integer
    )
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    batch_sz, B, vB, Iflat, Γ0 = _pseudolikelihood_context(rbm, v)
    site_linear = LinearIndices(size(rbm.visible))
    j = _on_device(Iflat, [site_linear[i] for i in reshape(sites, B)])
    b = _on_device(Iflat, collect(1:B))
    vflat = reshape(vB, length(rbm.visible), B)
    δ = ifelse.(vflat[CartesianIndex.(j, b)] .> 0, -flip, flip)
    Inew = Iflat .+ flat_w(rbm)[j, :]' .* reshape(δ, 1, B)
    Γ1 = vec(cgf(rbm.hidden, reshape(Inew, size(rbm.hidden)..., B)))
    ΔF = .-vec(rbm.visible.θ)[j] .* δ .- (Γ1 .- Γ0)
    return reshape(.-log1pexp.(.-ΔF), batch_sz)
end

function log_pseudolikelihood_sites(
        rbm::RBM{<:Binary}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
    )
    return _log_pseudolikelihood_sites_2states(rbm, v, sites, 1)
end

function log_pseudolikelihood_sites(
        rbm::RBM{<:Spin}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
    )
    return _log_pseudolikelihood_sites_2states(rbm, v, sites, 2)
end

function log_pseudolikelihood_sites(
        rbm::RBM{<:Potts}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
    )
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    @assert size(sites) == batch_size(rbm.visible, v)
    batch_sz, B, vB, Iflat, Γ0 = _pseudolikelihood_context(rbm, v)
    q = colors(rbm.visible)
    site_linear = LinearIndices(sitesize(rbm.visible))
    rows = _on_device(
        Iflat, [(site_linear[i] - 1) * q + x for x in 1:q, i in reshape(sites, B)]
    )
    b = _on_device(Iflat, collect(1:B))
    vflat = reshape(vB, length(rbm.visible), B)
    Vsite = vflat[CartesianIndex.(rows, reshape(b, 1, B))] # q × B, one-hot columns
    θsite = vec(rbm.visible.θ)[rows] # q × B
    Wsite = reshape(flat_w(rbm)[vec(rows), :], q, B, length(rbm.hidden)) # q × B × M
    Wold = reshape(sum(Wsite .* Vsite; dims = 1), B, length(rbm.hidden)) # B × M
    θold = vec(sum(θsite .* Vsite; dims = 1)) # B
    ΔF = similar(Γ0, q, B)
    for x in 1:q
        Inew = Iflat .+ (view(Wsite, x, :, :) .- Wold)'
        Γx = vec(cgf(rbm.hidden, reshape(Inew, size(rbm.hidden)..., B)))
        ΔF[x, :] .= θold .- view(θsite, x, :) .- (Γx .- Γ0)
    end
    lPL = -logsumexp(-ΔF; dims = 1)
    return reshape(lPL, batch_sz)
end

function log_pseudolikelihood_sites(
        rbm::RBM{<:PottsGumbel}, v::AbstractArray, sites::AbstractArray{<:CartesianIndex}
    )
    return log_pseudolikelihood_sites(RBM(Potts(rbm.visible), rbm.hidden, rbm.w), v, sites)
end

function _gaussian_hidden_pseudolikelihood_context(
        rbm::RBM{<:AbstractLayer, <:Gaussian}, v::AbstractArray
    )
    batch_sz = batch_size(rbm.visible, v)
    B = prod(batch_sz)
    vflat = reshape(v, length(rbm.visible), B)

    # The Gaussian cgf is quadratic. Its change under a hidden-input shift u is
    #
    #   Γ(z + u) - Γ(z) = u' * (z / |γ|) + u' * (u / |γ|) / 2.
    #
    # Precompute the terms shared by all visible-site substitutions. This turns
    # the repeated hidden cgf evaluations below into matrix multiplications.
    T = float(
        promote_type(
            eltype(rbm.visible.par), eltype(rbm.hidden.par), eltype(rbm.w)
        )
    )
    wflat = convert_eltype(T, flat_w(rbm))
    vcalc = convert_eltype(T, vflat)
    inputs = wflat' * vcalc
    θh = convert_eltype(T, vec(rbm.hidden.θ))
    γh = convert_eltype(T, vec(rbm.hidden.γ))
    invγ = inv.(abs.(γh))
    hidden_mean = (θh .+ inputs) .* reshape(invγ, length(rbm.hidden), 1)
    return batch_sz, B, vcalc, wflat, invγ, hidden_mean
end

function _log_pseudolikelihood_exact_2states(rbm::RBM, v::AbstractArray, flip::Integer)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_sz, B, vB, Iflat, Γ0 = _pseudolikelihood_context(rbm, v)
    vflat = reshape(vB, length(rbm.visible), B)
    wflat = flat_w(rbm)
    θflat = vec(rbm.visible.θ)
    lPL = zero(Γ0)
    δ = similar(Γ0)
    Inew = similar(Iflat)
    for j in 1:length(rbm.visible)
        δ .= ifelse.(view(vflat, j, :) .> 0, -flip, flip)
        Inew .= Iflat .+ view(wflat, j, :) .* reshape(δ, 1, B)
        Γ1 = vec(cgf(rbm.hidden, reshape(Inew, size(rbm.hidden)..., B)))
        Γ1 .-= Γ0
        lPL .-= log1pexp.(view(θflat, j:j) .* δ .+ Γ1)
    end
    lPL ./= length(rbm.visible)
    return reshape(lPL, batch_sz)
end

function _log_pseudolikelihood_exact_2states(
        rbm::RBM{<:Union{Binary, Spin}, <:Gaussian}, v::AbstractArray, flip::Integer
    )
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_sz, _, vflat, wflat, invγ, hidden_mean =
        _gaussian_hidden_pseudolikelihood_context(rbm, v)
    T = eltype(wflat)
    flipT = convert(T, flip)
    θv = reshape(convert_eltype(T, vec(rbm.visible.θ)), length(rbm.visible), 1)
    wscaled = wflat .* reshape(invγ, 1, length(rbm.hidden))
    linear = wflat * hidden_mean
    quadratic = vec(mapreduce(*, +, wscaled, wflat; dims = 2))
    quadratic ./= 2
    quadratic = reshape(quadratic, length(rbm.visible), 1)

    # If δ is the change of a visible unit, then
    # -ΔF = δ * (θv + W * (z / |γ|)) + δ² * diag(W / |γ| * W') / 2.
    minusΔF = linear
    @. minusΔF = (θv + minusΔF) * ifelse(vflat > 0, -flipT, flipT) +
        quadratic * abs2(flipT)
    lPL = vec(sum(.-log1pexp.(minusΔF); dims = 1))
    lPL ./= length(rbm.visible)
    return reshape(lPL, batch_sz)
end

function log_pseudolikelihood_exact(rbm::RBM{<:Binary}, v::AbstractArray)
    return _log_pseudolikelihood_exact_2states(rbm, v, 1)
end

function log_pseudolikelihood_exact(rbm::RBM{<:Spin}, v::AbstractArray)
    return _log_pseudolikelihood_exact_2states(rbm, v, 2)
end

function log_pseudolikelihood_exact(rbm::RBM{<:Potts}, v::AbstractArray)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_sz, B, vB, Iflat, Γ0 = _pseudolikelihood_context(rbm, v)
    q = colors(rbm.visible)
    nsites = prod(sitesize(rbm.visible))
    vflat = reshape(vB, length(rbm.visible), B)
    wflat = flat_w(rbm)
    θflat = vec(rbm.visible.θ)
    minusΔF = similar(Γ0, q, B)
    lPL = zero(Γ0)
    Inew = similar(Iflat)
    for j in 1:nsites
        rows = ((j - 1) * q + 1):(j * q)
        Vsite = with_eltype_of(wflat, view(vflat, rows, :)) # q × B, one-hot columns
        θsite = view(θflat, rows) # q
        Wsite = view(wflat, rows, :) # q × M
        Wold = Wsite' * Vsite # M × B
        θold = Vsite' * θsite # B
        for x in 1:q
            Inew .= Iflat .+ (view(Wsite, x, :) .- Wold)
            Γx = vec(cgf(rbm.hidden, reshape(Inew, size(rbm.hidden)..., B)))
            Γx .-= Γ0
            minusΔF[x, :] .= view(θsite, x:x) .- θold .+ Γx
        end
        lPL .-= vec(logsumexp(minusΔF; dims = 1))
    end
    lPL ./= nsites
    return reshape(lPL, batch_sz)
end

function log_pseudolikelihood_exact(
        rbm::RBM{<:Potts, <:Gaussian}, v::AbstractArray
    )
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    batch_sz, B, vflat, wflat, invγ, hidden_mean =
        _gaussian_hidden_pseudolikelihood_context(rbm, v)
    q = colors(rbm.visible)
    nsites = prod(sitesize(rbm.visible))
    linear = wflat * hidden_mean
    θflat = convert_eltype(eltype(linear), vec(rbm.visible.θ))
    lPL = zero(similar(linear, B))
    logits = similar(linear, q, B)
    wscaled_site = similar(linear, q, length(rbm.hidden))
    current = similar(linear, B)

    for j in 1:nsites
        rows = ((j - 1) * q + 1):(j * q)
        Vsite = view(vflat, rows, :)
        Wsite = view(wflat, rows, :)
        wscaled_site .= Wsite .* reshape(invγ, 1, length(rbm.hidden))
        gram = wscaled_site * Wsite'
        gram_v = gram * Vsite
        θsite = view(θflat, rows)
        linear_site = view(linear, rows, :)

        # Up to an x-independent term, -ΔF_x is the conditional logit
        #
        #   θ_x + w_x' * (z / |γ|) + w_x' * (w_x / |γ|) / 2
        #       - w_x' * (Wsite' * vsite / |γ|).
        #
        # Subtract x-independent terms before adding the visible fields. This
        # preserves small field differences when the hidden contribution has a
        # large common offset. Keep all three dense-input corrections explicit.
        current .= vec(sum(Vsite .* linear_site; dims = 1))
        logits .= linear_site .- reshape(current, 1, B)
        current .= vec(sum(Vsite .* reshape(θsite, q, 1); dims = 1))
        logits .+= reshape(θsite, q, 1) .- reshape(current, 1, B)
        current .= vec(sum(Vsite .* gram_v; dims = 1))
        logits .+= reshape(LinearAlgebra.diag(gram), q, 1) ./ 2 .- gram_v .+
            reshape(current, 1, B) ./ 2
        lPL .-= vec(logsumexp(logits; dims = 1))
    end
    lPL ./= nsites
    return reshape(lPL, batch_sz)
end

function log_pseudolikelihood_exact(rbm::RBM{<:PottsGumbel}, v::AbstractArray)
    return log_pseudolikelihood_exact(RBM(Potts(rbm.visible), rbm.hidden, rbm.w), v)
end
