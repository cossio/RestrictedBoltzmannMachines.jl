import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset, @inferred
using Statistics: mean
using LinearAlgebra: dot
using Random: bitrand, rand!
using Zygote: gradient, jacobian
using RestrictedBoltzmannMachines: visible, hidden, weights, free_energy, energy, interaction_energy
using RestrictedBoltzmannMachines: RBM, BinaryRBM, center_gradient, grad2mean, subtract_gradients
using RestrictedBoltzmannMachines: Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_h, ∂free_energy

struct CenteredRBM{V<:Binary,H<:Binary,W<:AbstractArray,Av<:AbstractArray,Ah<:AbstractArray}
    visible::V
    hidden::H
    w::W
    λv::Av
    λh::Ah
end

CenteredRBM(rbm::RBM, λv::Vector, λh::Vector) = CenteredRBM(
    visible(rbm), hidden(rbm), weights(rbm), λv, λh
)

CenteredRBM(a::Vector, b::Vector, w::Matrix, λv::Vector, λh::Vector) = CenteredRBM(
    BinaryRBM(a, b, w), λv, λh
)

RBM(rbm::CenteredRBM) = BinaryRBM(rbm.visible.θ, rbm.hidden.θ, rbm.w)

center(rbm::RBM{<:Binary,<:Binary}, λv::Vector, λh::Vector) = center(
    rbm.visible.θ, rbm.hidden.θ, rbm.w, λv, λh
)
center(a::Vector, b::Vector, w::Matrix, λv::Vector, λh::Vector) = CenteredRBM(
    a + w * λh, b + w' * λv, w, λv, λh
)
uncenter(rbm::CenteredRBM) = uncenter(rbm.visible.θ, rbm.hidden.θ, rbm.w, rbm.λv, rbm.λh)
uncenter(a::Vector, b::Vector, w::Matrix, λv::Vector, λh::Vector) = BinaryRBM(
    a - w * λh, b - w' * λv, w
)

function RBMs.energy(rbm::CenteredRBM, v::AbstractArray, h::AbstractArray)
    Ev = energy(rbm.visible, v)
    Eh = energy(rbm.hidden, h)
    Ew = interaction_energy(RBM(rbm), v .- rbm.λv, h .- rbm.λh)
    return Ev .+ Eh .+ Ew
end

function RBMs.free_energy(rbm::CenteredRBM, v::AbstractArray)
    inputs = RBMs.inputs_v_to_h(RBM(rbm), v .- rbm.λv)
    E = energy(rbm.visible, v) + energy(Binary(-rbm.λh), inputs)
    F = free_energy(rbm.hidden, inputs)
    return E + F
end

ΔE(rbm::CenteredRBM) = interaction_energy(RBM(rbm), rbm.λv, rbm.λh)

@testset "centered RBM" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    rbm_c = center(rbm, randn(3), randn(2))
    v = bitrand(3, 1000)
    h = bitrand(2, 1000)
    @test energy(rbm_c, v, h) ≈ energy(rbm, v, h) .+ ΔE(rbm_c)
    @test free_energy(rbm_c, v) ≈ free_energy(rbm, v) .+ ΔE(rbm_c)
    hs = [[0,0], [0,1], [1,0], [1,1]]
    @test exp.(-free_energy(rbm_c, v)) ≈ sum(exp.(-energy(rbm_c, v, h)) for h in hs)

    rbm_u = uncenter(rbm_c)
    @test rbm_u.visible.θ ≈ rbm.visible.θ
    @test rbm_u.hidden.θ ≈ rbm.hidden.θ
    @test rbm_u.w ≈ rbm.w
end

@testset "centered gradient" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    λv = randn(3)
    λh = randn(2)
    rbmc = center(rbm, λv, λh)
    v = bitrand(3, 1000)

    ∂ = subtract_gradients(∂free_energy(rbm, v), ∂free_energy(rbm, falses(3)))
    ∂c = @inferred center_gradient(rbm, ∂, rbmc.λv, rbmc.λh)

    ad, = gradient(rbmc) do rbmc
        mean(free_energy(rbmc, v)) - free_energy(rbmc, falses(3))
    end

    Ja, Jb, Jw = jacobian(rbmc.visible.θ, rbmc.hidden.θ, rbmc.w) do a, b, w
        uncenter(a, b, w, λv, λh).visible.θ
    end
    @test ∂c.visible.θ ≈ Ja * ad.visible.θ + Jb * ad.hidden.θ + Jw * vec(ad.w)

    Ja, Jb, Jw = jacobian(rbmc.visible.θ, rbmc.hidden.θ, rbmc.w) do a, b, w
        uncenter(a, b, w, λv, λh).hidden.θ
    end
    @test ∂c.hidden.θ ≈ Ja * ad.visible.θ + Jb * ad.hidden.θ + Jw * vec(ad.w)

    Ja, Jb, Jw = jacobian(rbmc.visible.θ, rbmc.hidden.θ, rbmc.w) do a, b, w
        uncenter(a, b, w, λv, λh).w
    end
    @test vec(∂c.w) ≈ Ja * ad.visible.θ + Jb * ad.hidden.θ + Jw * vec(ad.w)
end

"""
    struct2nt(obj)

Converts a struct to a NamedTuple with the same fields.
"""
struct2nt(s) = NamedTuple{propertynames(s)}(([getproperty(s, p) for p in propertynames(s)]...,))
@test struct2nt(1 + 2im) == (re = 1, im = 2)
@test struct2nt(Gaussian(ones(5), ones(5))) == (θ = ones(5), γ = ones(5))

_layers = (Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU)

@testset "centering $Layer gradients" for Layer in _layers
    layer = Layer(5)
    ∂ = struct2nt(layer)
    λ = randn(size(layer))
    applicable(center_gradient, layer, ∂, λ) || continue
    for p in propertynames(layer)
        rand!(getproperty(layer, p))
    end
    ∂c = center_gradient(layer, ∂, λ)
    layerc = deepcopy(layer)
    for p in propertynames(layer)
        getproperty(layerc, p) .= getproperty(∂c, p)
    end
    x = randn(size(layer)..., 10)
    @test energy(layer, x) ≈ energy(layerc, x) - x' * λ
end

@testset "grad2mean $Layer" for Layer in _layers
    layer = Layer(5)
    rbm = RBM(layer, Binary(randn(3)), randn(5,3))
    v = sample_v_from_v(rbm, randn(5,100); steps=100)
    ∂ = ∂free_energy(rbm, v)
    @test grad2mean(visible(rbm), ∂.visible) ≈ dropdims(mean(v; dims=2); dims=2)
end
