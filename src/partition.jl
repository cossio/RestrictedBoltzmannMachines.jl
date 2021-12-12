"""
    log_partition(rbm, β = 1)

Log-partition of the `rbm` at inverse temperature `β`, computed by extensive
enumeration of visible states
(except for particular cases such as Gaussian-Gaussian RBM).
This is exponentially slow for large machines.

If your RBM has a smaller hidden layer, consider using `flip_layers`.
"""
function log_partition(rbm::RBM, β::Real = 1)
    vs = iterate_states(rbm.visible)
    return LogExpFunctions.logsumexp(-β * only(free_energy(rbm, v, β)) for v in vs)
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
