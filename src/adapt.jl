# Teach Adapt.jl to recurse through our structs, so that `adapt(CuArray, rbm)`,
# `cu(rbm)`, `adapt(Array, rbm)`, and other array backends work out of the box.
Adapt.@adapt_structure Binary
Adapt.@adapt_structure Spin
Adapt.@adapt_structure Potts
Adapt.@adapt_structure Gaussian
Adapt.@adapt_structure ReLU
Adapt.@adapt_structure dReLU
Adapt.@adapt_structure pReLU
Adapt.@adapt_structure PottsGumbel
Adapt.@adapt_structure RBM
Adapt.@adapt_structure CenteredRBM
Adapt.@adapt_structure StandardizedRBM
Adapt.@adapt_structure ∂RBM

# manual method for xReLU (covers nsReLU) to preserve the FixGamma type parameter,
# which @adapt_structure would drop by calling the default xReLU(par) constructor
function Adapt.adapt_structure(to, l::xReLU{N,<:Any,FixGamma}) where {N,FixGamma}
    par = adapt(to, getfield(l, :par))
    return xReLU{N,typeof(par),FixGamma}(par)
end
