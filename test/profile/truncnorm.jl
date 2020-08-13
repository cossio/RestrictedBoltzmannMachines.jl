using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: truncnorm_quantile, tnrand
using Test, Random, Zygote


Random.seed!(1)
u = rand()
_, da, db = gradient(truncnorm_quantile, u, 0, 1)
Random.seed!(1)
da1, db1  = gradient(tnrand, 0, 1)
@test da1 ≈ da
@test db1 ≈ db

f(u,a,b) = sum(sin.(truncnorm_quantile.(u,a,b)))
g(a,b) = sum(sin.(tnrand.(a,b)))
Random.seed!(1)
u = rand(2,2)
_, da, db = gradient(f, u, zero(u), ones(u))
Random.seed!(1)
da1, db1 = gradient(g, zero(u), one(u))
@test da1 ≈ da
@test db1 ≈ db
