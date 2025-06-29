import Enzyme, Cthulhu, Zygote
using RestrictedBoltzmannMachines: Binary, Gaussian, RBM, standardize, unstandardize
using Test

Enzyme.Compiler.VERBOSE_ERRORS[] = false

rbm = standardize(RBM(Binary((3,)), Gaussian((2,)), randn(3,2)))
loss(rbm) = sum(unstandardize(rbm).w)

only(Enzyme.gradient(Enzyme.Reverse, loss, rbm))

Cthulhu.@descend_code_typed loss(rbm)
Cthulhu.@descend_code_typed standardize(rbm)

@code_typed standardize(rbm)

function fooz()
    rbm = standardize(RBM(Binary((3,)), Gaussian((2,)), randn(3,2)))
    gs = Zygote.gradient(rbm) do rbm
        sum(unstandardize(rbm).w)
    end
    return only(gs)
end
fooz()
