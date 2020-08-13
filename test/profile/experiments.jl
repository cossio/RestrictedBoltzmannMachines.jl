function binnext!(v::AbstractArray{Bool})
    for (i,a) in enumerate(v)
        if iszero(a)
            v[i] = true
            return v
        else
            v[i] = false
        end
    end
    v .= false
    return v
end


using Statistics, StructArrays, Zygote
struct Container{U,N,SA<:StructArray{U,N}} <: AbstractArray{U,N}
    units::SA
end
function Container{U}(fields::AbstractArray{<:Any,N}...) where {U,N}
    Layer(StructArray{U}(fields))
end
function f(x, y)
	l = Container{Complex}(x, y)
	return mean(l.units.x .+ l.units.y)
end
gradient(f, randn(2,2), randn(2,2))
