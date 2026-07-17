"""
    GaussianRBM(θv, γv, θh, γh, w)

Construct an RBM with Gaussian visible and hidden units.
Equivalent to `RBM(Gaussian(θv, γv), Gaussian(θh, γh), w)`.
"""
function GaussianRBM(θv::AbstractArray, γv::AbstractArray, θh::AbstractArray, γh::AbstractArray, w::AbstractArray)
    @assert size(θv) == size(γv)
    @assert size(θh) == size(γh)
    @assert size(w) == (size(θv)..., size(θh)...)
    return RBM(Gaussian(; θ = θv, γ = γv), Gaussian(; θ = θh, γ = γh), w)
end

function GaussianRBM(θv::AbstractArray, γv::AbstractArray, w::AbstractArray)
    @assert size(θv) == size(γv) == size(w)[1:ndims(θv)]
    θh = fill!(similar(θv, eltype(θv), size(w)[(ndims(θv) + 1):end]),  0)
    γh = fill!(similar(γv, eltype(γv), size(w)[(ndims(γv) + 1):end]),  1)
    return GaussianRBM(θv, γv, θh, γh, w)
end

# For a Gaussian-Gaussian RBM we can use the analytical expression
function log_partition(rbm::RBM{<:Gaussian, <:Gaussian})
    θ = [vec(rbm.visible.θ); vec(rbm.hidden.θ)]
    γv = vec(abs.(rbm.visible.γ))
    γh = vec(abs.(rbm.hidden.γ))
    w = flat_w(rbm)

    A = LinearAlgebra.Symmetric([
        Diagonal(γv) -w
        -w' Diagonal(γh)
    ])
    F = LinearAlgebra.cholesky(A; check = false)
    logZ0 = length(θ) / 2 * log(2π)

    if !LinearAlgebra.issuccess(F)
        return oftype(logZ0 + zero(eltype(F)) + zero(eltype(θ)), Inf)
    end

    return logZ0 + dot(θ, F \ θ) / 2 - logdet(F) / 2
end
