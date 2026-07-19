import Adapt
import ExplicitImports
import RestrictedBoltzmannMachines
using Test: @test, @testset

const RBMs = RestrictedBoltzmannMachines

fixture_path = joinpath(@__DIR__, "fixtures")
pushfirst!(LOAD_PATH, fixture_path)
try
    @eval import CUDA
finally
    @test popfirst!(LOAD_PATH) == fixture_path
end

@testset "CUDAExt ExplicitImports" begin
    extension_module = Base.get_extension(RBMs, :CUDAExt)
    @test !isnothing(extension_module)
    if !isnothing(extension_module)
        extension_file = pkgdir(RBMs, "ext", "CUDAExt.jl")
        # RBMs intentionally exports no symbols, so extension access to package
        # internals cannot satisfy a public-name check.
        from = (CUDA, Adapt)
        ExplicitImports.test_explicit_imports(
            extension_module,
            extension_file;
            all_explicit_imports_are_public = (; from),
            all_qualified_accesses_are_public = (; from),
        )
    end
end
