"""
    metropolis(rbm, v; β = 1, steps = 1)

Metropolis-Hastings sampling from `rbm` at inverse temperature `β`, starting from
configuration `v`. Moves are proposed by normal Gibbs sampling.
"""
function metropolis(rbm::RBM, v::AbstractArray; β::Real = 1, steps::Int = 1)
    @assert size(v)[1:ndims(rbm.visible)] == size(rbm.visible)
    for _ in 1:steps
        v = oftype(v, metropolis_once(rbm, v; β))
    end
    return v
end

function metropolis_once(rbm::RBM, v::AbstractArray; β::Real = 1)
    v_new = sample_v_from_v_once(rbm, v)
    ΔE = (β - 1) * (free_energy(rbm, v_new) - free_energy(rbm, v))
    sz = (ntuple(Returns(1), ndims(rbm.visible))..., batch_size(rbm.visible, v)...)
    α = exp.(-(ΔE isa Number ? ΔE : reshape(ΔE, sz))) # accept with probability min(1, α)
    u = rand!(similar(v, float(eltype(ΔE)), sz))
    return ifelse.(u .< α, v_new, eltype(v_new).(v)) # same eltype for a type-stable broadcast
end

"""
    metropolis!(v, rbm; β = 1)

Metropolis-Hastings sampling from `rbm` at inverse temperature β.
Uses `v[:,:,..,:,1]` as initial configurations, and writes the
Monte-Carlo chains in `v[:,:,..,:,2:end]`.
"""
function metropolis!(v::AbstractArray, rbm::RBM; β::Real = 1)
    @assert ndims(v) > ndims(rbm.visible) # last dim contains nsteps
    @assert size(v)[1:ndims(rbm.visible)] == size(rbm.visible)
    for t in 2:size(v, ndims(v))
        v[.., t] .= metropolis_once(rbm, v[.., t - 1]; β)
    end
    return v
end

"""
    cold_metropolis(rbm, v; steps = 1)

Samples the `rbm` at zero temperature, starting from configuration `v`.
"""
function cold_metropolis(rbm::RBM, v::AbstractArray; steps::Int = 1)
    @assert size(v)[1:ndims(rbm.visible)] == size(rbm.visible)
    for _ in 1:steps
        v = oftype(v, cold_metropolis_once(rbm, v))
    end
    return v
end

cold_metropolis_once(rbm::RBM, v::AbstractArray) = mode_v_from_h(rbm, mean_h_from_v(rbm, v))
