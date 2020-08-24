using Random, Test
using Zygote, FiniteDifferences
using Zygote: @adjoint
using RestrictedBoltzmannMachines: randn_like

function gradtest(f, args...)
    ftest(xs...) = sum(sin.(f(xs...)))
    us = 0.01 .* randn_like.(args)
    gs = gradient(ftest, args...)
    Δ = central_fdm(7,1)(ϵ -> ftest((args .+ ϵ .* us)...))
    @test Δ ≈ sum(sum(u .* g) for (u,g) in zip(us, gs))
end
