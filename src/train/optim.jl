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
