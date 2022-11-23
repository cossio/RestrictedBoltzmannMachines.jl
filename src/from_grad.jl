# get moments from layer cgf gradients, e.g. <v> = derivative of cgf w.r.t. θ

grad2ave(::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU}, ∂::AbstractArray) = ∂[1, ..]
grad2ave(::dReLU, ∂::AbstractArray) = ∂[1, ..] + ∂[2, ..]
