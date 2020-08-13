export log_pseudolikelihood, log_pseudolikelihood_rand

"""
    log_pseudolikelihood(sites, rbm, v, β=1)

Log-pseudolikelihood of a site conditioned on the other sites, where `sites`
is an array of site indices (CartesianIndex), one for each batch. Returns
an array of log-pseudolikelihood, for each batch.
"""
function log_pseudolikelihood(sites, rbm::RBM, v::AbstractArray, β=1)
    F = free_energy(rbm, v, β)
    F_ = log_site_traces(sites, rbm, v, β)
    return -β .* F - F_
end

"""
    log_pseudolikelihood(site, rbm, v, β=1)

Log-pseudolikelihood of a site conditioned on the other sites. Here `v` must
consist of a single batch.
"""
function log_pseudolikelihood(site::CartesianIndex, rbm::RBM, v::AbstractArray, β=1)
    size(rbm.vis) == size(v) || dimserror() # single batch
    F = free_energy(rbm, v, β)
    F_ = log_site_trace(site, rbm, v, β)
    return -β .* F - F_
end

"""
    log_site_traces(sites, rbm, v, β=1)

Log of the trace over configurations of `sites`, where `sites` is an array of
site indices (CartesianIndex), for each batch. Returns an array of the
log-traces for each batch.
"""
function log_site_traces(sites, rbm::RBM{<:Binary}, v::AbstractArray, β=1)
    bidx = batchindices(rbm.vis, v)
    F = free_energy(rbm, v, β)
    v_ = copy(v)
    for b in bidx
        v_[sites[b], b] = 1 - v_[sites[b], b]
    end
    F_ = free_energy(rbm, v_, β)
    logaddexp.(-β .* F, -β .* F_)
end

function log_site_traces(sites, rbm::RBM{<:Spin}, v::AbstractArray, β=1)
    bidx = batchindices(rbm.vis, v)
    F = free_energy(rbm, v, β)
    v_ = copy(v)
    for b in bidx
        v_[sites[b], b] = -v_[sites[b], b]
    end
    F_ = free_energy(rbm, v_, β)
    logaddexp.(-β .* F, -β .* F_)
end

function log_site_traces(sites, rbm::RBM{<:Potts}, v::AbstractArray, β=1)
    bidx = batchindices(rbm.vis, v)
    xidx = siteindices(rbm.vis)
    [log_site_trace(sites[b], rbm, v[:, xidx, b], β) for b in bidx]
end

"""
    log_site_trace(site, rbm, v, β=1)

Log of the trace over configurations of `site`. Here `v` must consist of
a single batch.
"""
function log_site_trace(site::CartesianIndex, rbm::RBM{<:Binary}, v::AbstractArray, β=1)
    size(rbm.vis) == size(v) || dimserror() # single batch
    v_ = copy(v)
    v_[site] = 1 - v_[site]
    logaddexp(-β * free_energy(rbm, v, β), -β * free_energy(rbm, v_, β))
end

function log_site_trace(site::CartesianIndex, rbm::RBM{<:Spin}, v::AbstractArray, β=1)
    size(rbm.vis) == size(v) || dimserror() # single batch
    v_ = copy(v)
    v_[site] = -v_[site]
    logaddexp(-β * free_energy(rbm, v, β), -β * free_energy(rbm, v_, β))
end

function log_site_trace(site::CartesianIndex, rbm::RBM{<:Potts}, v::AbstractArray, β=1)
    size(rbm.vis) == size(v) || dimserror() # single batch code
    v_ = copy(v)
    v_[:, site] .= false
    Fs = [free_energy_flip!(v_, site, a, rbm, β) for a in 1:rbm.vis.q]
    logsumexp(-β .* Fs)
end

function free_energy_flip!(v::AbstractArray, site::CartesianIndex, a::Int, rbm::RBM{<:Potts}, β=1)
    size(rbm.vis) == size(v) || dimserror() # single batch code
    v[a, site] = true
    F = free_energy(rbm, v, β)::Number
    v[a, site] = false
    return F
end

"""
    log_pseudolikelihood_full(rbm, v, β=1)

Average over all possible log-pseudolikelihoods (all sites).
This can be very slow.
"""
function log_pseudolikelihood_full(rbm::RBM, v::AbstractArray, β=1, w=1)
    xidx = siteindices(rbm.vis)
    if ndims(v) == ndims(rbm.vis)
        return mean(log_pseudolikelihood(site, rbm, v, β) for site in xidx)
    else
        bidx = batchindices(rbm.vis, v)
        S = (fill(site, size(bidx)) for site in xidx)
        return mean(wmean(log_pseudolikelihood(sites, rbm, v, β), w) for sites in S)
    end
end

"""
    log_pseudolikelihood_rand(rbm, v, β=1, w=1)

Log-pseudolikelihood of randomly chosen sites conditioned on the other sites.
For each configuration choses a random site, and returns the mean of the
computed pseudo-likelihoods.
"""
function log_pseudolikelihood_rand(rbm::RBM, v::AbstractArray, β=1, w=1)
    @assert ndims(v) > ndims(rbm.vis) # only for multiple batches
    xidx = siteindices(rbm.vis)
    bidx = batchindices(rbm.vis, v)
    sites = [rand(xidx) for b in bidx]
    return wmean(log_pseudolikelihood(sites, rbm, v, β), w)
end
