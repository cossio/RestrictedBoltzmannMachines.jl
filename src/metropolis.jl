"""
    metropolis!(v, rbm; β = 1)

Metropolis-Hastings sampling from `rbm` at inverse temperature β.
Uses `v[:,:,..,:,1]` as initial configurations, and writes the
Monte-Carlo chains in `v[:,:,..,:,2:end]`.
"""
function metropolis!(v::AbstractArray, rbm::RBM; β::Real = 1)
    nchains, nsteps = size(v)[end - 1], size(v)[end]
    @assert size(v) == (size(visible(rbm))..., nchains, nsteps)
    ℐ = CartesianIndices(size(visible(rbm))) # index span of visible layer
    for t in 2:nsteps
        v_old = selectdim(v, ndims(v), t - 1)
        v_new = sample_v_from_v_once(rbm, v_old)
        ΔE = (β - 1) * (free_energy(rbm, v_new) - free_energy(rbm, v_old))
        @assert length(ΔE) == nchains
        for n in 1:nchains
            if ΔE[n] ≤ 0 || randexp() > ΔE[n]
                # accept move
                v[ℐ, n, t] .= v_new[ℐ, n]
            else
                # do not accept move
                v[ℐ, n, t] .= v_old[ℐ, n]
            end
        end
    end
    return v
end
