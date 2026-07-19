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
    ExplicitImports.test_explicit_imports(
        RBMs;
        all_explicit_imports_are_public =
            (ignore = public_imports_without_legacy_metadata,),
        # Adapt documents @adapt_structure for package integration but does not
        # mark the macro public.
        all_qualified_accesses_are_public = (ignore = (Symbol("@adapt_structure"),),),
    )

    # The CUDA fixture uses CUDA's real UUID to load the actual package
    # extension. Keep it in a subprocess so it cannot trigger CUDA extensions
    # in unrelated test dependencies such as Zygote and NNlib.
    cuda_test = joinpath(@__DIR__, "explicit_imports_cuda.jl")
    project = dirname(Base.active_project())
    @test success(`$(Base.julia_cmd()) --project=$project $cuda_test`)

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
