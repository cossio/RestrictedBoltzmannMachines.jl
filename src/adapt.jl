# Teach Adapt.jl to recurse through our structs, so that `adapt(CuArray, rbm)`,
# `cu(rbm)`, `adapt(Array, rbm)`, and other array backends work out of the box.
for T in (
        Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, dReLU, pReLU, xReLU, nsReLU,
        RBM, CenteredRBM, StandardizedRBM, ∂RBM,
    )
    @eval Adapt.@adapt_structure $T
end
