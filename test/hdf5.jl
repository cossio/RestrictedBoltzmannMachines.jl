import HDF5
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: CenteredRBM
using RestrictedBoltzmannMachines: StandardizedRBM
using RestrictedBoltzmannMachines: Gaussian
using RestrictedBoltzmannMachines: load_rbm
using RestrictedBoltzmannMachines: Potts
using RestrictedBoltzmannMachines: PottsGumbel
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: ReLU
using RestrictedBoltzmannMachines: save_rbm
using RestrictedBoltzmannMachines: Spin
using RestrictedBoltzmannMachines: standardize
using RestrictedBoltzmannMachines: dReLU
using RestrictedBoltzmannMachines: pReLU
using RestrictedBoltzmannMachines: xReLU
using RestrictedBoltzmannMachines: nsReLU
using Test: @test
using Test: @testset
using Test: @test_throws

# every named parameter is a view into `par`, so comparing `par` covers them all
function check_roundtrip(rbm)
    loaded = load_rbm(save_rbm(tempname(), rbm))
    @test typeof(loaded.visible) === typeof(rbm.visible)
    @test typeof(loaded.hidden) === typeof(rbm.hidden)
    @test loaded.visible.par == rbm.visible.par
    @test loaded.hidden.par == rbm.hidden.par
    @test loaded.w == rbm.w
    return loaded
end

@testset "visible $(nameof(typeof(visible)))" for visible in (
        Binary(; θ = randn(2, 3)), Spin(; θ = randn(2, 3)),
        Potts(; θ = randn(2, 3)), PottsGumbel(; θ = randn(2, 3)),
    )
    check_roundtrip(RBM(visible, xReLU(; θ = randn(4), γ = randn(4), Δ = randn(4), ξ = randn(4)), randn(2, 3, 4)))
end

@testset "hidden $(nameof(typeof(hidden)))" for hidden in (
        Binary(; θ = randn(4)),
        Gaussian(; θ = randn(4), γ = randn(4)),
        ReLU(; θ = randn(4), γ = randn(4)),
        dReLU(; θp = randn(4), θn = randn(4), γp = randn(4), γn = randn(4)),
        pReLU(; θ = randn(4), γ = randn(4), Δ = randn(4), η = rand(4) .- 0.5),
        nsReLU(; θ = randn(4), Δ = randn(4), ξ = randn(4)),
    )
    check_roundtrip(RBM(Binary(; θ = randn(2, 3)), hidden, randn(2, 3, 4)))
end

@testset "centered" begin
    rbm = CenteredRBM(
        Binary(; θ = randn(2, 3)), ReLU(; θ = randn(4), γ = randn(4)), randn(2, 3, 4),
        randn(2, 3), randn(4),
    )
    loaded = check_roundtrip(rbm)
    @test loaded isa CenteredRBM
    @test loaded.offset_v == rbm.offset_v
    @test loaded.offset_h == rbm.offset_h
end

@testset "standardized" begin
    rbm = standardize(RBM(Binary(; θ = randn(2, 3)), xReLU(; θ = randn(4), γ = randn(4), Δ = randn(4), ξ = randn(4)), randn(2, 3, 4)))
    rbm.offset_v .= randn.()
    rbm.offset_h .= randn.()
    rbm.scale_v .= randn.()
    rbm.scale_h .= randn.()
    loaded = check_roundtrip(rbm)
    @test loaded isa StandardizedRBM
    @test loaded.offset_v == rbm.offset_v
    @test loaded.offset_h == rbm.offset_h
    @test loaded.scale_v == rbm.scale_v
    @test loaded.scale_h == rbm.scale_h
end

@testset "Float32 round-trip preserves eltype" begin
    rbm = RBM(
        Binary(; θ = randn(Float32, 2, 3)),
        Gaussian(; θ = randn(Float32, 4), γ = 1 .+ rand(Float32, 4)),
        randn(Float32, 2, 3, 4),
    )
    loaded = check_roundtrip(rbm)
    @test eltype(loaded.w) == eltype(loaded.visible.par) == eltype(loaded.hidden.par) == Float32
end

@testset "save_rbm refuses to overwrite" begin
    rbm = RBM(Binary(; θ = randn(3)), Binary(; θ = randn(2)), randn(3, 2))
    path = save_rbm(tempname(), rbm)
    @test_throws ErrorException save_rbm(path, rbm)
    @test save_rbm(path, rbm; overwrite = true) == path

    srbm = standardize(rbm)
    path = save_rbm(tempname(), srbm)
    @test_throws ErrorException save_rbm(path, srbm)
    @test save_rbm(path, srbm; overwrite = true) == path

    crbm = CenteredRBM(rbm)
    path = save_rbm(tempname(), crbm)
    @test_throws ErrorException save_rbm(path, crbm)
    @test save_rbm(path, crbm; overwrite = true) == path
end

@testset "load_rbm rejects unsupported format versions" begin
    rbm = RBM(Binary(; θ = randn(3)), Binary(; θ = randn(2)), randn(3, 2))
    path = save_rbm(tempname(), rbm)
    header = "rbm_hdf5_file_format_version"
    HDF5.h5open(path, "r+") do file
        HDF5.delete_object(file, header)
        HDF5.write(file, header, "0.0.0")
    end
    @test_throws ErrorException load_rbm(path)
end
