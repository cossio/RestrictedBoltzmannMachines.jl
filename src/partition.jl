"""
    log_partition(rbm, β = 1)

Log-partition of the `rbm` at inverse temperature `β`, computed by extensive
enumeration of  states. Only defined for discrete visible layers. This
is exponentially slow for large machines.
"""
function log_partition_from_visible(rbm::RBM, β::Real = 1)
    lZ = zero(free_energy(rbm, sample_from_inputs(rbm.visible), β))
    for v in iterate_states(rbm.visible)
        F = free_energy(rbm, reshape(v, size(rbm.visible)), β)
        lZ = logaddexp(lZ, -β * F)
    end
    return lZ
end

function log_partition(rbm::RBM{<:Gaussian, <:Gaussian}, β::Real = 1)
    W = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))
    ldet = sum(log, rbm.hidden.γ) + logdet(diagm(vec(rbm.visible.γ)) - W * diagm(1 ./ vec(rbm.hidden.γ)) * W')
    return (length(rbm.visible) + length(rbm.hidden)) / 2 * log(2π/β) - ldet / 2
end

alphabet(::Binary) = 0:1
alphabet(::Spin) = (-1,1)
alphabet(layer::Potts) = 1:layer.q

function iterate_states(layer::Bernoulli)
    itr = generate_sequences(length(layer.θ), 0:1)
    return map(x -> reshape(x, size(layer.θ)), itr)
end

"""
    log_likelihood(rbm, v, β=1)

Log-likelihood of `v` under `rbm`, with the partition function compued by
extensive enumeration. For discrete layers, this is exponentially slow for large machines.
"""
function log_likelihood(rbm::RBM, v::AbstractArray, β::Real = 1)
    lZ = log_partition(rbm, β)
    F = free_energy(rbm, v, β)
    ll = @. -β * F - lZ
    return ll ./ length(rbm.visible)
end

function mean_log_likelihood(rbm::RBM, v::AbstractArray, β::Real = 1, w = 1)
    return weighted_mean(log_likelihood(rbm, v, β), w)
end
