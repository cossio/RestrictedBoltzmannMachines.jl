import Aqua
import RestrictedBoltzmannMachines
using Test: @testset

@testset "aqua" begin
    Aqua.test_all(
        RestrictedBoltzmannMachines;
        stale_deps = (ignore = [:DiffRules],),
        ambiguities = (exclude = [reshape],),
        #project_toml_formatting = false
    )
end
