using Documenter, RestrictedBoltzmannMachines
import RestrictedBoltzmannMachines as RBMs

makedocs(
    modules=[RestrictedBoltzmannMachines],
    sitename="RestrictedBoltzmannMachines.jl"
)

deploydocs(
    repo = "github.com/cossio/RestrictedBoltzmannMachines.jl.git",
    devbranch = "master"
)
