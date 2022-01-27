include("tests_init.jl")
using InteractiveUtils: subtypes

function centered_energy(rbm, v, h; λv, λh)
    Ev = RBMs.energy(rbm.visible, v)
    Eh = RBMs.energy(rbm.hidden, h)
    return Ev .+ Eh .+ RBMs.interaction_energy(rbm, v .- λv, h .- λh)
end

function center(rbm; λv, λh)
    return RBMs.RBM(
        RBMs.Binary(rbm.visible.θ + rbm.w * λh),
        RBMs.Binary(rbm.hidden.θ + rbm.w' * λv),
        rbm.w
    )
end

@testset "centering binary RBMs" begin
    function energyc(rbm, v, h; λv, λh)
        @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)] == size(λv)
        @assert size(rbm.hidden)  == size(h)[1:ndims(rbm.hidden)]  == size(λh)
        Ev = RBMs.energy(rbm.visible, v)
        Eh = RBMs.energy(rbm.hidden, h)
        Ec = Ev .+ Eh .+ RBMs.interaction_energy(rbm, v .- λv, h .- λh)
        return Ec .- RBMs.interaction_energy(rbm, λv, λh)
    end

    function center(rbm; λv, λh)
        @assert size(rbm.visible) == size(λv)
        @assert size(rbm.hidden) == size(λh)
        return RBMs.RBM(
            RBMs.Binary(rbm.visible.θ + rbm.w * λh),
            RBMs.Binary(rbm.hidden.θ + rbm.w' * λv),
            rbm.w
        )
    end

    rbm = RBMs.RBM(RBMs.Binary(randn(4)), RBMs.Binary(randn(2)), randn(4,2))
    λv = randn(size(rbm.visible))
    λh = randn(size(rbm.hidden))
    v = bitrand(size(rbm.visible)...,3,2)
    h = bitrand(size(rbm.hidden)...,3,2)
    rbmc = center(rbm; λv, λh)
    @test center(rbmc; λv=-λv, λh=-λh).visible.θ ≈ rbm.visible.θ
    @test center(rbmc; λv=-λv, λh=-λh).hidden.θ ≈ rbm.hidden.θ
    @test center(rbmc; λv=-λv, λh=-λh).w ≈ rbm.w

    @test RBMs.interaction_energy(rbm, λv, λh) isa Number
    @test energyc(rbmc, v, h; λv, λh) ≈ RBMs.energy(rbm, v, h)
    ∂rbmc = only(Zygote.gradient(rbmc -> sum(energyc(rbmc, v, h; λv, λh)), rbmc))
    ∂rbm = only(Zygote.gradient(rbm -> sum(RBMs.energy(rbm, v, h)), rbm))
    ∂crbm = RBMs.center_gradients(rbm, ∂rbm, λv, λh);

    for i in eachindex(rbm.visible.θ)
        J = only(Zygote.gradient(rbmc -> center(rbmc; λv=-λv, λh=-λh).visible.θ[i], rbmc))
        @test ∂crbm.visible.θ[i] ≈ (
            (isnothing(J.visible) ? 0 : dot(J.visible.θ, ∂rbmc.visible.θ)) +
            (isnothing(J.hidden) ? 0 : dot(J.hidden.θ, ∂rbmc.hidden.θ)) +
            (isnothing(J.w) ? 0 : dot(J.w, ∂rbmc.w))
        )
    end

    for μ in eachindex(rbm.hidden.θ)
        J = only(Zygote.gradient(rbmc -> center(rbmc; λv=-λv, λh=-λh).hidden.θ[μ], rbmc))
        @test ∂crbm.hidden.θ[μ] ≈ (
            (isnothing(J.visible) ? 0 : dot(J.visible.θ, ∂rbmc.visible.θ)) +
            (isnothing(J.hidden) ? 0 : dot(J.hidden.θ, ∂rbmc.hidden.θ)) +
            (isnothing(J.w) ? 0 : dot(J.w, ∂rbmc.w))
        )
    end

    for k in eachindex(rbm.w)
        J = only(Zygote.gradient(rbmc -> center(rbmc; λv=-λv, λh=-λh).w[k], rbmc))
        @test ∂crbm.w[k] ≈ (
            (isnothing(J.visible) ? 0 : dot(J.visible.θ, ∂rbmc.visible.θ)) +
            (isnothing(J.hidden) ? 0 : dot(J.hidden.θ, ∂rbmc.hidden.θ)) +
            (isnothing(J.w) ? 0 : dot(J.w, ∂rbmc.w))
        )
    end
end

struct2nt(s) = NamedTuple{propertynames(s)}(([getproperty(s, p) for p in propertynames(s)]...,))

@testset "centering $Layer gradients" for Layer in subtypes(RBMs.AbstractLayer)
    layer = Layer(5)
    ∂ = struct2nt(layer)
    λ = randn(size(layer))
    applicable(RBMs.center_gradients, ∂, λ) || continue
    for p in propertynames(layer)
        rand!(getproperty(layer, p))
    end
    ∂c = RBMs.center_gradients(layer, ∂, λ)
    layerc = deepcopy(layer)
    for p in propertynames(layer)
        getproperty(layerc, p) .= getproperty(∂c, p)
    end
    x = randn(size(layer)..., 10)
    @test RBMs.energy(layer, x) ≈ RBMs.energy(layerc, x) - x' * λ
end
