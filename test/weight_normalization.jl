include("tests_init.jl")

@testset "∂wnorm" begin
    foo(w) = sum(sin.(w))
    function foo(g, v; N = count(size(g) .== 1))
        vnorm = sqrt.(sum(v .* v; dims=1:N))
        w = g .* v ./ vnorm
        return foo(w)
    end
    v = randn(5,3,4,2)
    g = randn(1,1,4,2)
    w = g .* v ./ sqrt.(sum!(similar(g), v .* v))
    ∂w, = Zygote.gradient(foo, w)
    ∂g, ∂v = Zygote.gradient(foo, g, v)
    ∂ = RBMs.∂wnorm(∂w, w, g, v)
    @test ∂.g ≈ ∂g
    @test ∂.v ≈ ∂v
end
