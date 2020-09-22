export log_likelihood, log_partition, mean_log_likelihood

"""
    log_partition(rbm, β = 1)

Log-partition of the `rbm` at inverse temperature `β`, computed by extensive
enumeration of visible states. Only defined for discrete visible layers. This
is exponentially slow for large machines.
"""
function log_partition(rbm::RBM, β = 1)
    lZ = zero(free_energy_v(rbm, random(rbm.vis), β))
    for v in seqgen(length(rbm.vis), alphabet(rbm.vis))
        F = free_energy_v(rbm, reshape(v, size(rbm.vis)), β)
        lZ = logaddexp(lZ, -β * F)
    end
    return lZ
end

alphabet(::Binary) = 0:1
alphabet(::Spin) = (-1,1)
alphabet(layer::Potts) = 1:layer.q

"""
    log_likelihood(rbm, v, β=1)

Log-likelihood of `v` under `rbm`, with the partition function compued by
extensive enumeration. Only defined for discrete visible layers. This is
exponentially slow for large machines.
"""
function log_likelihood(rbm::RBM, v::AbstractArray, β=1)
    lZ = log_partition(rbm, β)
    F = free_energy_v(rbm, v, β)
    ll = @. -β * F - lZ
    return ll ./ length(rbm.vis)
end

mean_log_likelihood(rbm::RBM, v::AbstractArray, β=1, w=1) =
    wmean(log_likelihood(rbm, v, β), w)
