import ExplicitImports
import RestrictedBoltzmannMachines
using Test: @test, @testset

const RBMs = RestrictedBoltzmannMachines

function load_hdf5_extension()
    @eval import HDF5
    return Base.get_extension(RBMs, :HDF5Ext)
end

@testset "ExplicitImports" begin
    # Julia 1.10 cannot represent `public` bindings. These documented APIs are
    # marked public by their owners on Julia 1.11+, where the ignore is empty.
    public_imports_without_legacy_metadata =
        VERSION < v"1.11" ? (:default_rng, :setup, :update!) : ()
    public_accesses_without_legacy_metadata = VERSION < v"1.11" ? (:tail,) : ()
    ExplicitImports.test_explicit_imports(
        RBMs;
        all_explicit_imports_are_public =
            (ignore = public_imports_without_legacy_metadata,),
        # Adapt documents @adapt_structure for package integration but does not
        # mark the macro public. Base documents @__doc__ as the way for macros
        # to attach docstrings to their expansions, but does not mark it public.
        all_qualified_accesses_are_public = (
            ignore = (
                Symbol("@adapt_structure"), Symbol("@__doc__"),
                public_accesses_without_legacy_metadata...,
            ),
        ),
    )

    # The CUDA fixture uses a non-CUDA UUID so unrelated CUDA extensions do not
    # mistake it for CUDA.jl; the child includes the actual CUDAExt source.
    # Keep it in a subprocess so its stub module remains isolated.
    cuda_test = joinpath(@__DIR__, "explicit_imports_cuda.jl")
    project = dirname(Base.active_project())
    process = run(ignorestatus(`$(Base.julia_cmd()) --project=$project $cuda_test`))
    @test process.exitcode == 0

    @testset "HDF5Ext" begin
        extension_module = load_hdf5_extension()
        @test !isnothing(extension_module)
        if !isnothing(extension_module)
            extension_file = pkgdir(RBMs, "ext", "HDF5Ext.jl")
            # RBMs intentionally exports no symbols, so extension access to
            # package internals cannot satisfy a public-name check.
            from = (getfield(@__MODULE__, :HDF5),)
            ExplicitImports.test_explicit_imports(
                extension_module,
                extension_file;
                all_explicit_imports_are_public = (; from),
                all_qualified_accesses_are_public = (; from),
            )
        end
    end
end
