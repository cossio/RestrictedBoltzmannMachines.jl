import HDF5
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: CenteredRBM
using RestrictedBoltzmannMachines: Gaussian
using RestrictedBoltzmannMachines: load_rbm
using RestrictedBoltzmannMachines: Potts
using RestrictedBoltzmannMachines: PottsGumbel
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: ReLU
using RestrictedBoltzmannMachines: save_rbm
using RestrictedBoltzmannMachines: Spin
using RestrictedBoltzmannMachines: standardize
using RestrictedBoltzmannMachines: StandardizedRBM
using RestrictedBoltzmannMachines: dReLU
using RestrictedBoltzmannMachines: pReLU
using RestrictedBoltzmannMachines: xReLU
using RestrictedBoltzmannMachines: nsReLU
using Test: @test
using Test: @testset
using Test: @test_throws

@testset "rbm" begin
    rbm = RBM(Potts(; θ=randn(2,3)), Binary(; θ=randn(4)), randn(2,3,4))
    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)
    @test loaded_rbm.visible isa Potts
    @test loaded_rbm.hidden isa Binary
    @test loaded_rbm.w == rbm.w
    @test loaded_rbm.visible.θ == rbm.visible.θ
    @test loaded_rbm.hidden.θ == rbm.hidden.θ
end

@testset "layers" begin
    for Layer = (Binary, Spin, Potts)
        rbm = RBM(Layer(; θ=randn(2,3)), xReLU(; θ=randn(4), γ=randn(4), Δ=randn(4), ξ=randn(4)), randn(2,3,4))
        path = save_rbm(tempname(), rbm)
        loaded_rbm = load_rbm(path)
        @test loaded_rbm.visible isa Layer
        @test loaded_rbm.hidden isa xReLU
        @test loaded_rbm.w == rbm.w
        @test loaded_rbm.visible.θ == rbm.visible.θ
        @test loaded_rbm.hidden.θ == rbm.hidden.θ
        @test loaded_rbm.hidden.γ == rbm.hidden.γ
        @test loaded_rbm.hidden.Δ == rbm.hidden.Δ
        @test loaded_rbm.hidden.ξ == rbm.hidden.ξ
    end
end

@testset "Gaussian" begin
    rbm = RBM(Binary(; θ=randn(2,3)), Gaussian(; θ=randn(4), γ=randn(4)), randn(2,3,4))
    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)
    @test loaded_rbm.visible isa Binary
    @test loaded_rbm.hidden isa Gaussian
    @test loaded_rbm.w == rbm.w
    @test loaded_rbm.visible.θ == rbm.visible.θ
    @test loaded_rbm.hidden.θ == rbm.hidden.θ
    @test loaded_rbm.hidden.γ == rbm.hidden.γ
end

@testset "nsReLU" begin
    rbm = RBM(Binary(; θ=randn(2,3)), nsReLU(; θ=randn(4), Δ=randn(4), ξ=randn(4)), randn(2,3,4))
    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)
    @test loaded_rbm.visible isa Binary
    @test loaded_rbm.hidden isa nsReLU
    @test loaded_rbm.w == rbm.w
    @test loaded_rbm.visible.θ == rbm.visible.θ
    @test loaded_rbm.hidden.θ == rbm.hidden.θ
    @test loaded_rbm.hidden.Δ == rbm.hidden.Δ
    @test loaded_rbm.hidden.ξ == rbm.hidden.ξ
end

@testset "PottsGumbel" begin
    rbm = RBM(PottsGumbel(; θ=randn(2,3)), Binary(; θ=randn(4)), randn(2,3,4))
    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)
    @test loaded_rbm.visible isa PottsGumbel
    @test loaded_rbm.w == rbm.w
    @test loaded_rbm.visible.θ == rbm.visible.θ
    @test loaded_rbm.hidden.θ == rbm.hidden.θ
end

@testset "ReLU" begin
    rbm = RBM(Binary(; θ=randn(2,3)), ReLU(; θ=randn(4), γ=randn(4)), randn(2,3,4))
    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)
    @test loaded_rbm.visible isa Binary
    @test loaded_rbm.hidden isa ReLU
    @test loaded_rbm.w == rbm.w
    @test loaded_rbm.visible.θ == rbm.visible.θ
    @test loaded_rbm.hidden.θ == rbm.hidden.θ
    @test loaded_rbm.hidden.γ == rbm.hidden.γ
end

@testset "dReLU" begin
    rbm = RBM(
        Binary(; θ=randn(2,3)),
        dReLU(; θp=randn(4), θn=randn(4), γp=randn(4), γn=randn(4)),
        randn(2,3,4),
    )
    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)
    @test loaded_rbm.visible isa Binary
    @test loaded_rbm.hidden isa dReLU
    @test loaded_rbm.w == rbm.w
    @test loaded_rbm.visible.θ == rbm.visible.θ
    @test loaded_rbm.hidden.θp == rbm.hidden.θp
    @test loaded_rbm.hidden.θn == rbm.hidden.θn
    @test loaded_rbm.hidden.γp == rbm.hidden.γp
    @test loaded_rbm.hidden.γn == rbm.hidden.γn
end

@testset "pReLU" begin
    rbm = RBM(
        Binary(; θ=randn(2,3)),
        pReLU(; θ=randn(4), γ=randn(4), Δ=randn(4), η=randn(4)),
        randn(2,3,4),
    )
    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)
    @test loaded_rbm.visible isa Binary
    @test loaded_rbm.hidden isa pReLU
    @test loaded_rbm.w == rbm.w
    @test loaded_rbm.visible.θ == rbm.visible.θ
    @test loaded_rbm.hidden.θ == rbm.hidden.θ
    @test loaded_rbm.hidden.γ == rbm.hidden.γ
    @test loaded_rbm.hidden.Δ == rbm.hidden.Δ
    @test loaded_rbm.hidden.η == rbm.hidden.η
end

@testset "centered" begin
    rbm = CenteredRBM(
        Binary(; θ=randn(2,3)),
        ReLU(; θ=randn(4), γ=randn(4)),
        randn(2,3,4),
        randn(2,3),
        randn(4),
    )
    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)

    @test loaded_rbm.visible isa Binary
    @test loaded_rbm.hidden isa ReLU
    @test loaded_rbm.w == rbm.w
    @test loaded_rbm.visible.θ == rbm.visible.θ
    @test loaded_rbm.hidden.θ == rbm.hidden.θ
    @test loaded_rbm.hidden.γ == rbm.hidden.γ
    @test loaded_rbm.offset_v == rbm.offset_v
    @test loaded_rbm.offset_h == rbm.offset_h
end

@testset "std" begin
    rbm = standardize(RBM(Binary(; θ=randn(2,3)), xReLU(; θ=randn(4), γ=randn(4), Δ=randn(4), ξ=randn(4)), randn(2,3,4)))
    rbm.offset_v .= randn.()
    rbm.offset_h .= randn.()
    rbm.scale_v .= randn.()
    rbm.scale_h .= randn.()

    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)

    @test loaded_rbm.visible isa Binary
    @test loaded_rbm.hidden isa xReLU

    @test loaded_rbm.w == rbm.w

    @test loaded_rbm.visible.θ == rbm.visible.θ

    @test loaded_rbm.hidden.θ == rbm.hidden.θ
    @test loaded_rbm.hidden.γ == rbm.hidden.γ
    @test loaded_rbm.hidden.Δ == rbm.hidden.Δ
    @test loaded_rbm.hidden.ξ == rbm.hidden.ξ

    @test loaded_rbm.offset_v == rbm.offset_v
    @test loaded_rbm.offset_h == rbm.offset_h
    @test loaded_rbm.scale_v == rbm.scale_v
    @test loaded_rbm.scale_h == rbm.scale_h
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

@testset "Float32 round-trip preserves eltype" begin
    rbm = RBM(
        Binary(; θ = randn(Float32, 2, 3)),
        Gaussian(; θ = randn(Float32, 4), γ = 1 .+ rand(Float32, 4)),
        randn(Float32, 2, 3, 4),
    )
    path = save_rbm(tempname(), rbm)
    loaded_rbm = load_rbm(path)
    @test eltype(loaded_rbm.w) == Float32
    @test eltype(loaded_rbm.visible.par) == Float32
    @test eltype(loaded_rbm.hidden.par) == Float32
    @test loaded_rbm.w == rbm.w
    @test loaded_rbm.visible.θ == rbm.visible.θ
    @test loaded_rbm.hidden.θ == rbm.hidden.θ
    @test loaded_rbm.hidden.γ == rbm.hidden.γ
end
