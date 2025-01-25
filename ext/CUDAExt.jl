module CUDAExt

import RestrictedBoltzmannMachines
using CUDA: cu
using Adapt: adapt
using RestrictedBoltzmannMachines: gpu, cpu
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: Spin
using RestrictedBoltzmannMachines: Potts
using RestrictedBoltzmannMachines: Gaussian
using RestrictedBoltzmannMachines: ReLU
using RestrictedBoltzmannMachines: dReLU
using RestrictedBoltzmannMachines: pReLU
using RestrictedBoltzmannMachines: xReLU
using RestrictedBoltzmannMachines: PottsGumbel
using RestrictedBoltzmannMachines: ∂RBM

RestrictedBoltzmannMachines.gpu(x::AbstractArray) = cu(x)
RestrictedBoltzmannMachines.cpu(x::AbstractArray) = adapt(Array, x)

RestrictedBoltzmannMachines.gpu(rbm::RBM) = RBM(gpu(rbm.visible), gpu(rbm.hidden), gpu(rbm.w))
RestrictedBoltzmannMachines.cpu(rbm::RBM) = RBM(cpu(rbm.visible), cpu(rbm.hidden), cpu(rbm.w))
RestrictedBoltzmannMachines.gpu(∂::∂RBM) = ∂RBM(gpu(∂.visible), gpu(∂.hidden), gpu(∂.w))
RestrictedBoltzmannMachines.cpu(∂::∂RBM) = ∂RBM(cpu(∂.visible), cpu(∂.hidden), cpu(∂.w))
RestrictedBoltzmannMachines.gpu(layer::Binary) = Binary(gpu(layer.par))
RestrictedBoltzmannMachines.cpu(layer::Binary) = Binary(cpu(layer.par))
RestrictedBoltzmannMachines.gpu(layer::Spin) = Spin(gpu(layer.par))
RestrictedBoltzmannMachines.cpu(layer::Spin) = Spin(cpu(layer.par))
RestrictedBoltzmannMachines.gpu(layer::Potts) = Potts(gpu(layer.par))
RestrictedBoltzmannMachines.cpu(layer::Potts) = Potts(cpu(layer.par))
RestrictedBoltzmannMachines.gpu(layer::Gaussian) = Gaussian(gpu(layer.par))
RestrictedBoltzmannMachines.cpu(layer::Gaussian) = Gaussian(cpu(layer.par))
RestrictedBoltzmannMachines.gpu(layer::ReLU) = ReLU(gpu(layer.par))
RestrictedBoltzmannMachines.cpu(layer::ReLU) = ReLU(cpu(layer.par))
RestrictedBoltzmannMachines.gpu(layer::dReLU) = dReLU(gpu(layer.par))
RestrictedBoltzmannMachines.cpu(layer::dReLU) = dReLU(cpu(layer.par))
RestrictedBoltzmannMachines.gpu(layer::pReLU) = pReLU(gpu(layer.par))
RestrictedBoltzmannMachines.cpu(layer::pReLU) = pReLU(cpu(layer.par))
RestrictedBoltzmannMachines.gpu(layer::xReLU) = xReLU(gpu(layer.par))
RestrictedBoltzmannMachines.cpu(layer::xReLU) = xReLU(cpu(layer.par))
RestrictedBoltzmannMachines.gpu(layer::PottsGumbel) = PottsGumbel(gpu(layer.par))
RestrictedBoltzmannMachines.cpu(layer::PottsGumbel) = PottsGumbel(cpu(layer.par))

end
