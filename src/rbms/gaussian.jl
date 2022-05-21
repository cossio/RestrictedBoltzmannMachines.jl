function GaussianRBM(θv::AbstractArray, γv::AbstractArray, θh::AbstractArray, γh::AbstractArray, w::AbstractArray)
    @assert size(θv) == size(γv)
    @assert size(θh) == size(γh)
    @assert size(w) == (size(θv)..., size(θh)...)
    return RBM(Gaussian(θv, γv), Gaussian(θh, γh), w)
end

function GaussianRBM(θv::AbstractArray, γv::AbstractArray, w::AbstractArray)
    @assert size(θv) == size(γv) == size(w)[1:ndims(θv)]
    θh = fill!(similar(θv, eltype(θv), size(w)[(ndims(θv) + 1):end]),  0)
    γh = fill!(similar(γv, eltype(γv), size(w)[(ndims(γv) + 1):end]),  1)
    return GaussianRBM(θv, γv, θh, γh, w)
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
