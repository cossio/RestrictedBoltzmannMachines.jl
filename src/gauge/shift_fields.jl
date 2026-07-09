# Functions to add a constant to the fields of a layer. This is useful in many situations,
# such as zerosum, CenteredRBM, and so on.

# the fields θ are the leading row(s) of `par`; other rows are preserved
function shift_fields(l::Union{Binary,Spin,Potts,PottsGumbel,Gaussian,ReLU,pReLU,xReLU}, a::AbstractArray)
    par = copy(getfield(l, :par))
    par[1, ..] .+= a
    return typeof(l)(par)
end

function shift_fields(l::dReLU, a::AbstractArray)
    par = copy(getfield(l, :par))
    par[1, ..] .+= a
    par[2, ..] .+= a
    return typeof(l)(par)
end

function shift_fields!(l::Union{Binary,Spin,Potts,PottsGumbel,Gaussian,ReLU,pReLU,xReLU}, a::AbstractArray)
    l.θ .+= a
    return l
end

function shift_fields!(l::dReLU, a::AbstractArray)
    l.θp .+= a
    l.θn .+= a
    return l
end
