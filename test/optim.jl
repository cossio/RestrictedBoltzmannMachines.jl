using Test: @test, @testset, @inferred
using RestrictedBoltzmannMachines: ∂RBM

@testset "∂ operations" begin
    ∂1 = ∂RBM(randn(1,3), randn(1,2), randn(3,2))
    ∂2 = ∂RBM(randn(1,3), randn(1,2), randn(3,2))
    @test @inferred(∂1 + ∂2) == ∂RBM(∂1.visible + ∂2.visible, ∂1.hidden + ∂2.hidden, ∂1.w + ∂2.w)
    @test @inferred(∂1 - ∂2) == ∂RBM(∂1.visible - ∂2.visible, ∂1.hidden - ∂2.hidden, ∂1.w - ∂2.w)
    @test @inferred(2 * ∂1) == ∂RBM(2 * ∂1.visible, 2 * ∂1.hidden, 2 * ∂1.w)
    @test @inferred(∂1 / 2) == ∂RBM(∂1.visible / 2, ∂1.hidden / 2, ∂1.w / 2)
end

@testset "∂RBM hash and equality" begin
    ∂1 = ∂RBM(randn(1,3), randn(1,2), randn(3,2))
    ∂2 = ∂RBM(copy(∂1.visible), copy(∂1.hidden), copy(∂1.w))
    @test ∂1 == ∂2
    @test hash(∂1) == hash(∂2)
    ∂3 = ∂RBM(randn(1,3), randn(1,2), randn(3,2))
    @test ∂1 != ∂3
end
