module CUDAExt

import RestrictedBoltzmannMachines
using CUDA: cu
using Adapt: adapt

# Layers, RBMs and ∂RBM define `Adapt.adapt_structure` (see src/adapt.jl), so `cu`
# and `adapt` recurse through them and these two methods cover arrays and structs alike.
RestrictedBoltzmannMachines.gpu(x) = cu(x)
RestrictedBoltzmannMachines.cpu(x) = adapt(Array, x)

end
