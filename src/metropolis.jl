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
    @assert size(v)[1:ndims(rbm.visible)] == size(rbm.visible)
    v_new = sample_v_from_v_once(rbm, v)
    ΔE = (β - 1) * (free_energy(rbm, v_new) - free_energy(rbm, v))
    ℐ = CartesianIndices(size(rbm.visible)) # index span of visible layer
    for n in CartesianIndices(ΔE)
        if ΔE[n] ≤ 0 || ΔE[n] < randexp()
            continue # accept move
        else
            v_new[ℐ, n] .= v[ℐ, n] # do not accept move
        end
    end
    return v_new
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

function cold_metropolis_once(rbm::RBM, v::AbstractArray)
    @assert size(v)[1:ndims(rbm.visible)] == size(rbm.visible)
    h = mean_h_from_v(rbm, v)
    return mode_v_from_h(rbm, h)
end
