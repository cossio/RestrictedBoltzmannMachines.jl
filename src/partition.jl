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
    return LogExpFunctions.logsumexp(-β * free_energy(rbm, v, β) for v in vs)
end

# For a Gaussian-Gaussian RBM we can use the analytical expression
function log_partition(rbm::RBM{<:Gaussian, <:Gaussian}, β::Real = 1)
    N, M = length(rbm.visible), length(rbm.hidden)
    θv = Diagonal(vec(β * abs.(rbm.visible.θ)))
    θh = Diagonal(vec(β * abs.(rbm.hidden.θ)))
    γv = Diagonal(vec(β * abs.(rbm.visible.γ)))
    γh = Diagonal(vec(β * abs.(rbm.hidden.γ)))
    w = reshape(β * rbm.weights, N, M)
    lA = block_matrix_logdet(
        γv, w,
        w', γh
    )
    iA = block_matrix_invert(
        γv, w,
        w', γh
    )
    return (N + M)/2 * log(2π) + [θv' θh'] * iA * [θv; θh] / 2 - lA/2
end

"""
    log_likelihood(rbm, v, β=1)

Log-likelihood of `v` under `rbm`, with the partition function compued by
extensive enumeration. For discrete layers, this is exponentially slow for large machines.
"""
function log_likelihood(rbm::RBM, v::AbstractArray, β::Real = true)
    lZ = log_partition(rbm, β)
    F = free_energy(rbm, v, β)
    return -β .* F .- lZ
end

function iterate_states(layer::Binary)
    itr = generate_sequences(length(layer), 0:1)
    return map(x -> reshape(x, size(layer), 1), itr)
end

function iterate_states(layer::Spin)
    itr = generate_sequences(length(layer), (-1,1))
    return map(x -> reshape(x, size(layer), 1), itr)
end

function iterate_states(layer::Potts)
    error("not implemented")
end
