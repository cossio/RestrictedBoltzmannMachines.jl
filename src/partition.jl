"""
    log_partition(rbm)

Log-partition of `rbm`, computed by extensive enumeration of visible states
(except for particular cases such as Gaussian-Gaussian RBM).
This is exponentially slow for large machines.

If your RBM has a smaller hidden layer, mirroring the layers of the `rbm` first
(see [`mirror`](@ref)).
"""
function log_partition(rbm::RBM)
    v = ChainRulesCore.ignore_derivatives() do
        collect_states(visible(rbm))
    end
    return logsumexp(-free_energy(rbm, v))
end

# For a Gaussian-Gaussian RBM we can use the analytical expression
function log_partition(rbm::RBM{<:Gaussian, <:Gaussian})
    θ = [vec(visible(rbm).θ); vec(hidden(rbm).θ)]
    γv = vec(abs.(visible(rbm).γ))
    γh = vec(abs.(hidden(rbm).γ))
    w = flat_w(rbm)

    lA = block_matrix_logdet(
        Diagonal(γv), -w,
        -w', Diagonal(γh)
    )

    iA = block_matrix_invert(
        Diagonal(γv), -w,
        -w', Diagonal(γh)
    )

    return length(θ)/2 * log(2π) + dot(θ, iA, θ)/2 - lA/2
end

"""
    log_likelihood(rbm, v)

Log-likelihood of `v` under `rbm`, with the partition function compued by
extensive enumeration. For discrete layers, this is exponentially slow for large machines.
"""
function log_likelihood(rbm::RBM, v::AbstractArray)
    logZ = log_partition(rbm)
    F = free_energy(rbm, v)
    return -F .- logZ
end

function iterate_states(layer::Binary)
    itr = generate_sequences(length(layer), false:true)
    return map(x -> reshape(x, size(layer)..., 1), itr)
end

function iterate_states(layer::Spin)
    itr = generate_sequences(length(layer), (-Int8(1), Int8(1)))
    return map(x -> reshape(x, size(layer)..., 1), itr)
end

function iterate_states(::Potts)
    error("not implemented")
end

"""
    collect_states(layer)

Returns an array of all states of `layer`.
Only defined for discrete layers.

!!! warning
    Use only for small layers.
    For large layers, the exponential number of states will not fit in memory.
"""
function collect_states(layer::Union{Binary, Spin, Potts})
    return cat(iterate_states(layer)...; dims=ndims(layer) + 1)
end
