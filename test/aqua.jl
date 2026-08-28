import Aqua
import RestrictedBoltzmannMachines
using Test: @testset

# Work around a Julia 1.12 parallel-precompilation race (extension build-id
# invalidation, e.g. LogExpFunctionsChainRulesCoreExt) that makes the wrapper
# package in Aqua.test_persistent_tasks fail to precompile softly whenever its
# freshly-resolved dependency graph is cold: Pkg files the wrapper under
# "failed but may be precompilable after restarting julia" and exits 0, so
# Aqua's done.log sentinel never appears. Precompiling an equivalent
# environment once beforehand absorbs the race, so Aqua's wrapper compiles
# against warm caches. No-op when caches are already warm.
function prewarm_persistent_tasks_env(pkg::Module)
    dir = mktempdir()
    code = """
    push!(LOAD_PATH, "@stdlib")
    using Pkg
    Pkg.activate($(repr(dir)); io = devnull)
    Pkg.develop(Pkg.PackageSpec(path = $(repr(pkgdir(pkg)))); io = devnull)
    Pkg.precompile()
    """
    for _ in 1:2
        success(`$(Base.julia_cmd()) --startup-file=no -e $code`) && break
    end
    return nothing
end

prewarm_persistent_tasks_env(RestrictedBoltzmannMachines)

@testset "aqua" begin
    Aqua.test_all(
        RestrictedBoltzmannMachines;
        stale_deps = (ignore = [:Adapt],),
        ambiguities = (exclude = [reshape],),
        #project_toml_formatting = false
    )
end
