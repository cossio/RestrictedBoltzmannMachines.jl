import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset, @inferred
using Statistics: mean
using LinearAlgebra: dot
using Random: bitrand, rand!
using Zygote: gradient, jacobian
using RestrictedBoltzmannMachines: free_energy, energy, interaction_energy,
    RBM, BinaryRBM, grad2ave, center_gradient, centered_gradient,
    Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU, uncenter_step,
    sample_v_from_v, sample_h_from_h, ∂free_energy, ∂cgf, cgf, inputs_h_from_v, ∂RBM

struct CenteredRBM{V,H,W,Av,Ah}
    visible::V
    hidden::H
    w::W
    λv::Av
    λh::Ah
    function CenteredRBM(
        visible::Binary, hidden::Binary, w::AbstractArray,
        λv::AbstractArray, λh::AbstractArray
    )
        @assert size(w) == (size(visible)..., size(hidden)...)
        @assert size(λv) == size(visible)
        @assert size(λh) == size(hidden)
        return new{typeof(visible), typeof(hidden), typeof(w), typeof(λv), typeof(λh)}(
            visible, hidden, w, λv, λh
        )
    end
end

function CenteredRBM(rbm::RBM, λv::AbstractVector, λh::AbstractVector)
    return CenteredRBM(rbm.visible, rbm.hidden, rbm.w, λv, λh)
end

function CenteredRBM(
    a::AbstractVector, b::AbstractVector, w::AbstractMatrix,
    λv::AbstractVector, λh::AbstractVector
)
    rbm = BinaryRBM(a, b, w)
    return CenteredRBM(rbm, λv, λh)
end

RBMs.RBM(rbm::CenteredRBM) = BinaryRBM(rbm.visible.θ, rbm.hidden.θ, rbm.w)

function center(rbm::RBM, λv::AbstractVector, λh::AbstractVector)
    return center(rbm.visible.θ, rbm.hidden.θ, rbm.w, λv, λh)
end

function center(
    a::AbstractVector, b::AbstractVector, w::AbstractMatrix,
    λv::AbstractVector, λh::AbstractVector
)
    ac = a + w * λh
    bc = b + w' * λv
    return CenteredRBM(ac, bc, w, λv, λh)
end

uncenter(rbm::CenteredRBM) = uncenter(rbm.visible.θ, rbm.hidden.θ, rbm.w, rbm.λv, rbm.λh)

function uncenter(
    a::AbstractVector, b::AbstractVector, w::AbstractMatrix,
    λv::AbstractVector, λh::AbstractVector
)
    return BinaryRBM(a - w * λh, b - w' * λv, w)
end

function RBMs.energy(rbm::CenteredRBM, v::AbstractArray, h::AbstractArray)
    Ev = energy(rbm.visible, v)
    Eh = energy(rbm.hidden, h)
    Ew = interaction_energy(RBM(rbm), v .- rbm.λv, h .- rbm.λh)
    return Ev .+ Eh .+ Ew
end

function RBMs.free_energy(rbm::CenteredRBM, v::AbstractArray)
    inputs = inputs_h_from_v(RBM(rbm), v .- rbm.λv)
    E = energy(rbm.visible, v) + energy(Binary(; θ = -rbm.λh), inputs)
    Γ = cgf(rbm.hidden, inputs)
    return E - Γ
end

ΔE(rbm::CenteredRBM) = interaction_energy(RBM(rbm), rbm.λv, rbm.λh)

@testset "centered RBM" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    rbm_c = @inferred center(rbm, randn(3), randn(2))
    v = bitrand(3, 100)
    h = bitrand(2, 100)
    @test energy(rbm_c, v, h) ≈ energy(rbm, v, h) .+ ΔE(rbm_c)
    @test free_energy(rbm_c, v) ≈ free_energy(rbm, v) .+ ΔE(rbm_c)
    hs = [[0,0], [0,1], [1,0], [1,1]]
    @test exp.(-free_energy(rbm_c, v)) ≈ sum(exp.(-energy(rbm_c, v, h)) for h in hs)

    rbm_u = @inferred uncenter(rbm_c)
    @test rbm_u.visible.θ ≈ rbm.visible.θ
    @test rbm_u.hidden.θ ≈ rbm.hidden.θ
    @test rbm_u.w ≈ rbm.w
end

@testset "center_gradient" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    λv = randn(3)
    λh = randn(2)
    rbmc = center(rbm, λv, λh)
    v = bitrand(3, 1000)

    # subtraction eliminates constant energy displacement
    ∂, = gradient(rbm) do rbm
        mean(free_energy(rbm, v)) - free_energy(rbm, falses(3))
    end
    ∂ = @inferred ∂RBM(∂.visible.par, ∂.hidden.par, ∂.w)
    ∂c = @inferred center_gradient(rbm, ∂, λv, λh)

    gs, = gradient(rbmc) do rbmc
        mean(free_energy(rbmc, v)) - free_energy(rbmc, falses(3))
    end
    @test ∂c.visible ≈ gs.visible.par
    @test ∂c.hidden ≈ gs.hidden.par
    @test ∂c.w ≈ gs.w
end

@testset "centered gradient" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    λv = randn(3)
    λh = randn(2)
    rbmc = center(rbm, λv, λh)
    v = bitrand(3, 1000)

    # subtraction eliminates constant energy displacement
    ∂ = ∂free_energy(rbm, v) - ∂free_energy(rbm, falses(3))
    ∂c = @inferred centered_gradient(rbm, ∂, rbmc.λv, rbmc.λh)

    ad, = gradient(rbmc) do rbmc
        mean(free_energy(rbmc, v)) - free_energy(rbmc, falses(3))
    end

    # since the centering transformation is linear, we can just uncenter the gradient
    ad_u = uncenter(dropdims(ad.visible.par; dims=1), dropdims(ad.hidden.par; dims=1), ad.w, λv, λh)

    @test ∂c.visible ≈ ad_u.visible.par
    @test ∂c.hidden ≈ ad_u.hidden.par
    @test ∂c.w ≈ ad_u.w
end

@testset "centered gradient via jacobian" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    λv = randn(3)
    λh = randn(2)
    rbmc = center(rbm, λv, λh)
    v = bitrand(3, 1000)

    # subtraction eliminates constant energy displacement
    ∂ = ∂free_energy(rbm, v) - ∂free_energy(rbm, falses(3))
    ∂c = @inferred centered_gradient(rbm, ∂, rbmc.λv, rbmc.λh)

    ad, = gradient(rbmc) do rbmc
        mean(free_energy(rbmc, v)) - free_energy(rbmc, falses(3))
    end

    Ja, Jb, Jw = jacobian(rbmc.visible.θ, rbmc.hidden.θ, rbmc.w) do a, b, w
        uncenter(a, b, w, λv, λh).visible.θ
    end
    @test ∂c.visible' ≈ Ja * ad.visible.par' + Jb * ad.hidden.par' + Jw * vec(ad.w)

    Ja, Jb, Jw = jacobian(rbmc.visible.θ, rbmc.hidden.θ, rbmc.w) do a, b, w
        uncenter(a, b, w, λv, λh).hidden.θ
    end
    @test ∂c.hidden' ≈ Ja * ad.visible.par' + Jb * ad.hidden.par' + Jw * vec(ad.w)

    Ja, Jb, Jw = jacobian(rbmc.visible.θ, rbmc.hidden.θ, rbmc.w) do a, b, w
        uncenter(a, b, w, λv, λh).w
    end
    @test vec(∂c.w) ≈ Ja * ad.visible.par' + Jb * ad.hidden.par' + Jw * vec(ad.w)
end

_layers = (Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU)

@testset "shifting $Layer gradients" for Layer in _layers
    layer = Layer((5,))
    ∂ = layer.par
    λ = randn(size(layer))
    applicable(uncenter_step, layer, ∂, λ) || continue
    for p in propertynames(layer)
        rand!(getproperty(layer, p))
    end
    ∂c = uncenter_step(layer, ∂, λ)
    layerc = deepcopy(layer)
    layerc.par .= ∂c
    x = randn(size(layer)..., 10)
    @test energy(layer, x) ≈ energy(layerc, x) - x' * λ
end

@testset "grad2ave $Layer" for Layer in _layers
    layer = Layer((5,))
    rbm = RBM(layer, Binary(; θ = randn(3)), randn(5,3))
    v = sample_v_from_v(rbm, randn(5,100); steps=100)
    ∂ = ∂free_energy(rbm, v)
    @test (@inferred grad2ave(rbm.visible, -∂.visible)) ≈ dropdims(mean(v; dims=2); dims=2)
end
