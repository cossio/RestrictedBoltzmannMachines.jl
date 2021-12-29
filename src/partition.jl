"""
    log_partition(rbm, β = 1)

Log-partition of the `rbm` at inverse temperature `β`, computed by extensive
enumeration of visible states
(except for particular cases such as Gaussian-Gaussian RBM).
This is exponentially slow for large machines.

If your RBM has a smaller hidden layer, consider using `flip_layers`.
"""
function log_partition(rbm::RBM, β::Real = true)
    v = ChainRulesCore.ignore_derivatives() do
        collect_states(rbm.visible)
    end
    return LogExpFunctions.logsumexp(-β * free_energy(rbm, v, β))
end

# For a Gaussian-Gaussian RBM we can use the analytical expression
function log_partition(rbm::RBM{<:Gaussian, <:Gaussian}, β::Real = true)
    θ = β * [vec(rbm.visible.θ); vec(rbm.hidden.θ)]
    γv = β * vec(abs.(rbm.visible.γ))
    γh = β * vec(abs.(rbm.hidden.γ))
    w = β * reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))

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
    log_likelihood(rbm, v, β=1)

Log-likelihood of `v` under `rbm`, with the partition function compued by
extensive enumeration. For discrete layers, this is exponentially slow for large machines.
"""
function log_likelihood(rbm::RBM, v::AbstractArray, β::Real = true)
    logZ = log_partition(rbm, β)
    F = free_energy(rbm, v, β)
    return -β .* F .- logZ
end

function iterate_states(layer::Binary)
    itr = generate_sequences(length(layer), 0:1)
    return map(x -> reshape(x, size(layer)..., 1), itr)
end

function iterate_states(layer::Spin)
    itr = generate_sequences(length(layer), (-1,1))
    return map(x -> reshape(x, size(layer)..., 1), itr)
end

function iterate_states(layer::Potts)
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
