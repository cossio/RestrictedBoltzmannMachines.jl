#=
Zygote uses ForwardDiff for differentiating broadcasting operations.
See https://github.com/FluxML/Zygote.jl/pull/1001.
However ForwardDiff cannot differentiate non-generic code that accepts only Float64
(it needs to pass its Dual type).
In addition ForwardDiff defines rules via DiffRules. It doesn't understand ChainRules.
This means we need to define some rules here to make things work.

See discussion here to understand how to add new rules for ForwardDiff:
https://discourse.julialang.org/t/issue-with-forwarddiff-custom-ad-rule/72886/6

If https://github.com/JuliaDiff/DiffRules.jl/pull/74 gets merged I should not need
this anymore.
=#

import IrrationalConstants, SpecialFunctions, DiffRules
import ForwardDiff # must be last import

# logerfcx
∂logerfcx(x) = 2 * (x - inv(SpecialFunctions.erfcx(x)) / IrrationalConstants.sqrtπ)
DiffRules.@define_diffrule SpecialFunctions.logerfcx(x) = :(∂logerfcx($x))
eval(ForwardDiff.unary_dual_definition(:SpecialFunctions, :logerfcx))
