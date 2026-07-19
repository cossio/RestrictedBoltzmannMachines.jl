import Adapt
import ExplicitImports
import RestrictedBoltzmannMachines
using Test: @test, @testset

const RBMs = RestrictedBoltzmannMachines

fixture_path = joinpath(@__DIR__, "fixtures")
pushfirst!(LOAD_PATH, fixture_path)
try
    @eval import CUDA

    @testset "CUDAExt ExplicitImports" begin
        extension_file = pkgdir(RBMs, "ext", "CUDAExt.jl")
        @test isnothing(Base.get_extension(RBMs, :CUDAExt))
        extension_module = include(extension_file)
        @test extension_module === CUDAExt

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
finally
    @test popfirst!(LOAD_PATH) == fixture_path
end
