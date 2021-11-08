"""
    log_partition(rbm, β = 1)

Log-partition of the `rbm` at inverse temperature `β`, computed by extensive
enumeration of  states (except for particular cases such as Gaussian-Gaussian)
RBM). This is exponentially slow for large machines.
"""
function log_partition(rbm::RBM, β::Real = 1)
    v = cat(iterate_states(rbm.visible)...; dims=ndims(rbm.visible) + 1)
    @assert size(v) == (size(rbm.visible)..., size(v)[end])
    F = free_energy(rbm, v, β)
    return logsumexp(-β * F)
end

# For a Gaussian-Gaussian RBM we have an analytical expression
function log_partition(rbm::RBM{<:Gaussian, <:Gaussian}, β::Real = 1)
    N, M = length(rbm.visible), length(rbm.hidden)
    Γv = Diagonal(vec(β * rbm.visible.γ))
    Γh = Diagonal(vec(β * rbm.hidden.γ))
    W = reshape(β * rbm.weights, N, M)
    ldet = logdet(Γv) + logdet(Γh - W' * inv(Γv) * W)
    return (N + M)/2 * log(2π) - ldet/2
end

"""
    log_likelihood(rbm, v, β=1)

Log-likelihood of `v` under `rbm`, with the partition function compued by
extensive enumeration. For discrete layers, this is exponentially slow for large machines.
"""
function log_likelihood(rbm::RBM, v::AbstractArray, β::Real = 1)
    lZ = log_partition(rbm, β)
    F = free_energy(rbm, v, β)
    ll = -β .* F .- lZ
    return ll
end


function iterate_states(layer::Binary)
    itr = generate_sequences(length(layer.θ), 0:1)
    return map(x -> reshape(x, size(layer.θ)), itr)
end

function iterate_states(layer::Spin)
    itr = generate_sequences(length(layer.θ), (-1, 1))
    return map(x -> reshape(x, size(layer.θ)), itr)
end

function iterate_states(layer::Potts)
    error("not implemented")
end
