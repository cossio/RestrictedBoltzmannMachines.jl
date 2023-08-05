import Aqua
import RestrictedBoltzmannMachines

using Test: @testset

@testset "aqua" begin
    Aqua.test_all(
        RestrictedBoltzmannMachines;
        stale_deps=(ignore=[:DiffRules],),
        ambiguities=(recursive=false, exclude=[reshape])
    )
end
