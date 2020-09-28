ignore(f) = error("Attempt to call ignore outside Zygote")
@adjoint ignore(f) = f(), _ -> nothing
macro ignore(ex)
    return :(ignore() do
        $(esc(ex))
    end)
end

# # logerfcx calls erf which are not pure Julia code.
# @adjoint function SpecialFunctions.logerfcx(x::Real)
#     y = logerfcx(x)
#     dx = 2x - 2 / exp(y) / √oftype(y,π)
#     return y, δ -> (δ * dx,)
# end

@adjoint function broadcasted(::typeof(logerfcx), x::Num)
    y = logerfcx.(x)
    PI = convert(eltype(x), π)
    dx = @. 2x - 2/exp(y)/√PI
    y, δ -> (nothing, δ .* dx)
end

@adjoint function broadcasted(::typeof(log1pexp), x::Num)
    y = log1pexp.(x)
    D = NNlib.sigmoid.(x)
    y, δ -> (nothing, δ .* D)
end

@adjoint function StatsFuns.logaddexp(x::Real, y::Real)
    result, dx, dy = ∇logaddexp(x, y)
    back(δ) = (δ * dx, δ * dy)
    return result, back
end
@adjoint function broadcasted(::typeof(logaddexp), x::Num, y::Num)
    result, dx, dy = ∇logaddexp(x, y)
    back(δ) = (nothing, unbroadcast(x, δ .* dx), unbroadcast(y, δ .* dy))
    return result, back
end
function ∇logaddexp(x::Num, y::Num)
    result = logaddexp.(x, y)
    t = @. exp(-abs(x - y))
    dx, dy = select(x .≥ y, inv.(one.(t) .+ t), t ./ (one.(t) .+ t))
    return result, dx, dy
end

@adjoint function broadcasted(::typeof(sqrt), x::Num)
    result = sqrt.(x)
    result, δ -> (nothing, δ ./ result ./ 2)
end

@adjoint function sumdrop(xs::NumArray; dims)
    S = sum(xs; dims=dims)
	back(δ::NumArray) = (similar(xs) .= reshape(δ, size(S)...),)
	back(δ::Number) = (similar(xs) .= δ,)
    dropdims(S; dims=dims), back
end

@adjoint function sumdropfirst(A::NumArray, ::Val{N}) where {N}
	dims = OneHot.tuplen(Val(N))
	S = sum(A; dims=dims)
	back(δ::NumArray) = (similar(A) .= reshape(δ, size(S)...), nothing)
	back(δ::Number) = (similar(A) .= δ, nothing)
	dropdims(S; dims=dims), back
end
