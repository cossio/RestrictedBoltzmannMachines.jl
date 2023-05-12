struct ∂RBM{V,H,W}
    visible::V
    hidden::H
    w::W
    function ∂RBM(visible::AbstractArray, hidden::AbstractArray, w::AbstractArray)
        @assert size(w) == (tail(size(visible))..., tail(size(hidden))...)
        return new{typeof(visible), typeof(hidden), typeof(w)}(visible, hidden, w)
    end
end

Base.:(+)(∂1::∂RBM, ∂2::∂RBM) = ∂RBM(∂1.visible + ∂2.visible, ∂1.hidden + ∂2.hidden, ∂1.w + ∂2.w)
Base.:(-)(∂1::∂RBM, ∂2::∂RBM) = ∂RBM(∂1.visible - ∂2.visible, ∂1.hidden - ∂2.hidden, ∂1.w - ∂2.w)
Base.:(*)(λ::Real, ∂::∂RBM) = ∂RBM(λ * ∂.visible, λ * ∂.hidden, λ * ∂.w)
Base.:(*)(∂::∂RBM, λ::Real) = λ * ∂
Base.:(/)(∂::∂RBM, λ::Real) = ∂RBM(∂.visible / λ, ∂.hidden / λ, ∂.w / λ)
Base.:(==)(∂1::∂RBM, ∂2::∂RBM) = (∂1.visible == ∂2.visible) && (∂1.hidden == ∂2.hidden) && (∂1.w == ∂2.w)
Base.hash(∂::∂RBM, h::UInt) = hash(∂.visible, hash(∂.hidden, hash(∂.w, h)))

"""
    ∂free_energy(rbm, v)

Gradient of `free_energy(rbm, v)` with respect to model parameters.
If `v` consists of multiple samples (batches), then an average is taken.
"""
function ∂free_energy(
    rbm::RBM, v::AbstractArray; wts = nothing,
    moments = moments_from_samples(rbm.visible, v; wts)
)
    inputs = inputs_h_from_v(rbm, v)
    ∂v = ∂energy_from_moments(rbm.visible, moments)
    ∂Γ = ∂cgfs(rbm.hidden, inputs)
    h = grad2ave(rbm.hidden, ∂Γ)
    ∂h = reshape(wmean(-∂Γ; wts, dims = (ndims(rbm.hidden.par) + 1):ndims(∂Γ)), size(rbm.hidden.par))
    ∂w = ∂interaction_energy(rbm, v, h; wts)
    return ∂RBM(∂v, ∂h, ∂w)
end

function ∂interaction_energy(rbm::RBM, v::AbstractArray, h::AbstractArray; wts=nothing)
    bsz = batch_size(rbm, v, h)
    if ndims(rbm.visible) == ndims(v) && ndims(rbm.hidden) == ndims(h)
        wts::Nothing
        vflat = with_eltype_of(rbm.w, vec(v))
        hflat = with_eltype_of(rbm.w, vec(h))
        ∂wflat = -vflat * hflat'
    elseif ndims(rbm.visible) == ndims(v)
        vflat = with_eltype_of(rbm.w, vec(v))
        hflat = with_eltype_of(rbm.w, vec(batchmean(rbm.hidden, h; wts)))
        ∂wflat = -vflat * hflat'
    elseif ndims(rbm.hidden) == ndims(h)
        vflat = with_eltype_of(rbm.w, vec(batchmean(rbm.visible, v; wts)))
        hflat = with_eltype_of(rbm.w, vec(h))
        ∂wflat = -vflat * hflat'
    else
        vflat = with_eltype_of(rbm.w, flatten(rbm.visible, v))
        hflat = with_eltype_of(rbm.w, flatten(rbm.hidden, h))
        if isnothing(wts)
            ∂wflat = -vflat * hflat' / size(vflat, 2)
        else
            @assert size(wts) == bsz
            ∂wflat = -vflat * Diagonal(vec(wts)) * hflat' / sum(wts)
        end
    end
    ∂w = reshape(∂wflat, size(rbm.w))
    return ∂w
end
