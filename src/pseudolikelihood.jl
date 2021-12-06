"""
    log_pseudolikelihood_rand(rbm, v, β=1)

Log-pseudolikelihood of randomly chosen sites conditioned on the other sites.
For each configuration choses a sample_from_inputs site, and returns the mean of the
computed pseudo-likelihoods.
"""
function log_pseudolikelihood(rbm::RBM, v::AbstractArray, β::Real = true)
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    xidx = CartesianIndices(size(rbm.visible))
    sites = [rand(xidx) for b in 1:_nobs(v)]
    return log_pseudolikelihood(rbm, v, sites, β)
end

"""
    log_pseudolikelihood(rbm, v, sites, β=1)

Log-pseudolikelihood of a site conditioned on the other sites, where `sites`
is an array of site indices (CartesianIndex), one for each batch. Returns
an array of log-pseudolikelihood, for each batch.
"""
function log_pseudolikelihood(
    rbm::RBM,
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex},
    β::Real = true
)
    F = free_energy(rbm, v, β)
    F_ = log_site_traces(rbm, v, sites, β)
    return -β .* F - F_
end

"""
    log_site_traces(rbm, v, sites, β=1)

Log of the trace over configurations of `sites`, where `sites` is an array of
site indices (CartesianIndex), for each batch. Returns an array of the
log-traces for each batch.
"""
function log_site_traces(
    rbm::RBM,
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex},
    β::Real = true
)
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    F = free_energy(rbm, v, β)
    v_ = copy(v)
    for b in 1:_nobs(v)
        v_[sites[b], b] = 1 - v_[sites[b], b]
    end
    F_ = free_energy(rbm, v_, β)
    return LogExpFunctions.logaddexp.(-β .* F, -β .* F_)
end

function log_site_traces(
    rbm::RBM{<:Spin},
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex},
    β::Real = 1
)
    bidx = batchindices(rbm.visible, v)
    F = free_energy(rbm, v, β)
    v_ = copy(v)
    for b in bidx
        v_[sites[b], b] = -v_[sites[b], b]
    end
    F_ = free_energy(rbm, v_, β)
    return LogExpFunctions.logaddexp.(-β .* F, -β .* F_)
end

function log_site_traces(
    rbm::RBM{<:Potts},
    v::AbstractArray,
    sites::AbstractVector{<:CartesianIndex},
    β::Real = 1
)
    bidx = batchindices(rbm.visible, v)
    xidx = siteindices(rbm.visible)
    [log_site_trace(rbm, v[:, xidx, b], sites[b], β) for b in bidx]
end

"""
    log_site_trace(site, rbm, v, β=1)

Log of the trace over configurations of `site`. Here `v` must consist of
a single batch.
"""
function log_site_trace(rbm::RBM{<:Binary}, v::AbstractArray, site::CartesianIndex, β::Real = true)
    size(rbm.visible) == size(v) || dimserror() # single batch
    v_ = copy(v)
    v_[site] = 1 - v_[site]
    F = free_energy(rbm, v, β)
    F_ = free_energy(rbm, v_, β)
    return LogExpFunctions.logaddexp(-β * F, -β * F_)
end

function log_site_trace(site::CartesianIndex, rbm::RBM{<:Spin}, v::AbstractArray, β::Real = true)
    size(rbm.visible) == size(v) || dimserror() # single batch
    v_ = copy(v)
    v_[site] = -v_[site]
    F = free_energy(rbm, v, β)
    F_ = free_energy(rbm, v_, β)
    return LogExpFunctions.logaddexp(-β * F, -β * F_)
end

function log_site_trace(site::CartesianIndex, rbm::RBM{<:Potts}, v::AbstractArray, β::Real = 1)
    size(rbm.visible) == size(v) || dimserror() # single batch code
    v_ = copy(v)
    v_[:, site] .= false
    Fs = [free_energy_flip!(v_, site, a, rbm, β) for a in 1:rbm.visible.q]
    return LogExpFunctions.logsumexp(-β .* Fs)
end

function free_energy_flip!(v::AbstractArray, site::CartesianIndex, a::Int, rbm::RBM{<:Potts}, β::Real = 1)
    size(rbm.visible) == size(v) || dimserror() # single batch code
    v[a, site] = true
    F = free_energy(rbm, v, β)::Number
    v[a, site] = false
    return F
end
