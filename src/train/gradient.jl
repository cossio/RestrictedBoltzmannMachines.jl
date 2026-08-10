struct ∂RBM{V, H, W}
    visible::V
    hidden::H
    w::W
    function ∂RBM(visible::AbstractArray, hidden::AbstractArray, w::AbstractArray)
        @assert size(w) == (size(visible)[2:end]..., size(hidden)[2:end]...)
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
        rbm, v::AbstractArray; wts::AbstractArray{<:Real} = uniform_wts(rbm.visible, v),
        moments = moments_from_samples(rbm.visible, v; wts)
    )
    inputs = inputs_h_from_v(rbm, v)
    h_moments = moments_from_inputs(rbm.hidden, inputs)
    # both layer gradients are ∂energy_from_moments: at the data moments for the
    # visible layer, at the conditional moments <.|v> for the hidden layer
    ∂v = ∂energy_from_moments(rbm.visible, moments)
    ∂h = ∂energy_from_moments(rbm.hidden, batchmean_moments(rbm.hidden, h_moments; wts))
    h = mean_from_moments(rbm.hidden, h_moments)
    ∂w = ∂interaction_energy(rbm, v, h; wts)
    return ∂RBM(∂v, ∂h, ∂w)
end

∂free_energy_v(rbm, v::AbstractArray; kwargs...) = ∂free_energy(rbm, v; kwargs...)

function ∂free_energy_h(
        rbm, h::AbstractArray; wts::AbstractArray{<:Real} = uniform_wts(rbm.hidden, h),
        moments = moments_from_samples(rbm.hidden, h; wts)
    )
    inputs = inputs_v_from_h(rbm, h)
    v_moments = moments_from_inputs(rbm.visible, inputs)
    ∂v = ∂energy_from_moments(rbm.visible, batchmean_moments(rbm.visible, v_moments; wts))
    ∂h = ∂energy_from_moments(rbm.hidden, moments)
    v = mean_from_moments(rbm.visible, v_moments)
    ∂w = ∂interaction_energy(rbm, v, h; wts)
    return ∂RBM(∂v, ∂h, ∂w)
end

function ∂interaction_energy(
        rbm::RBM, v::AbstractArray, h::AbstractArray;
        wts::AbstractArray{<:Real} = Trues(batch_size(rbm, v, h))
    )
    bsz = batch_size(rbm, v, h)
    @assert size(wts) == bsz
    if ndims(rbm.visible) == ndims(v) && ndims(rbm.hidden) == ndims(h)
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
        # weighted batch average as a Diagonal-weighted matmul, as in `batchcov`
        vflat = with_eltype_of(rbm.w, flatten(rbm.visible, v))
        hflat = with_eltype_of(rbm.w, flatten(rbm.hidden, h))
        ∂wflat = -_weighted_outer(vflat, wts, hflat) / sum(wts)
    end
    ∂w = reshape(∂wflat, size(rbm.w))
    return ∂w
end
