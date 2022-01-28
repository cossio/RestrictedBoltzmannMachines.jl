include("tests_init.jl")

@testset "∂wnorm" begin
    foo(w) = sum(sin.(w))
    function foo(g, v; N = count(size(g) .== 1))
        vnorm = sqrt.(sum(v .* v; dims=1:N))
        w = g.^2 .* v ./ vnorm
        return foo(w)
    end
    g = randn(1,1,4,2)
    v = randn(5,3,4,2)
    w = abs2.(g) .* v ./ sqrt.(sum!(similar(g), abs2.(v)))
    ∂w, = Zygote.gradient(foo, w)
    ∂g, ∂v = Zygote.gradient(foo, g, v)
    ∂ = RBMs.∂wnorm(∂w, w, g, v)
    @test ∂.g ≈ ∂g
    @test ∂.v ≈ ∂v

    randn!(v)
    randn!(g)
    RBMs.update_w_from_gv!(w, g, v)
    @test w ≈ abs2.(g) .* v ./ sqrt.(sum!(similar(g), abs2.(v)))

    rbm = RBMs.BinaryRBM(randn(5), randn(3), randn(5,3))
    @test RBMs.weight_norms(rbm) ≈ sqrt.(sum(abs2.(rbm.w); dims=1))

    wn = RBMs.WeightNorm(rbm)
    @test abs2.(wn.g) ≈ RBMs.weight_norms(rbm)
    @test wn.v ./ sqrt.(sum(abs2.(wn.v); dims=1)) ≈ rbm.w ./ sqrt.(sum(abs2.(rbm.w); dims=1))
end
