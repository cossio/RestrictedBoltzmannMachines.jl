export log_likelihood, log_partition, mean_log_likelihood

"""
    log_partition(rbm, β = 1)

Log-partition of the `rbm` at inverse temperature `β`, computed by extensive
enumeration of visible states. Only defined for discrete visible layers. This
is exponentially slow for large machines.
"""
function log_partition(rbm::RBM{V,H,W}, β::Num = 1) where {V<:AbstractDiscreteLayer,H,W}
    lZ = zero(free_energy_v(rbm, random(rbm.vis), β))
    for v in seqgen(length(rbm.vis), alphabet(rbm.vis))
        F = free_energy_v(rbm, reshape(v, size(rbm.vis)), β)
        lZ = logaddexp(lZ, -β * F)
    end
    return lZ
end

function log_partition(rbm::RBM{<:Gaussian, <:Gaussian}, β::Num = 1)
    W = reshape(rbm.weights, length(rbm.vis), length(rbm.hid))
    ldet = sum(log, rbm.hid.γ) + logdet(diagm(vec(rbm.vis.γ)) - W * diagm(1 ./ vec(rbm.hid.γ)) * W')
    return (length(rbm.vis) + length(rbm.hid)) / 2 * log(2π/β) - ldet / 2
end

alphabet(::Binary) = 0:1
alphabet(::Spin) = (-1,1)
alphabet(layer::Potts) = 1:layer.q

"""
    log_likelihood(rbm, v, β=1)

Log-likelihood of `v` under `rbm`, with the partition function compued by
extensive enumeration. For discrete layers, this is
exponentially slow for large machines.
"""
function log_likelihood(rbm::RBM, v::NumArray, β::Num = 1)
    lZ = log_partition(rbm, β)
    F = free_energy_v(rbm, v, β)
    ll = @. -β * F - lZ
    return ll ./ length(rbm.vis)
end

mean_log_likelihood(rbm::RBM, v::NumArray, β::Num = 1, w::Num = 1) =
    wmean(log_likelihood(rbm, v, β), w)
