import RestrictedBoltzmannMachines as RBMs # make sure the rules we defined are loaded
using SpecialFunctions
using ForwardDiff: derivative

@testset "∂ logerfcx" begin
    @test derivative(logerfcx,  1) ≈ -0.638967514234791260471501152
    @test derivative(logerfcx, -1) ≈ -2.225271242628657455193006951
    @test derivative(logerfcx,  2) ≈ -0.418160805994414197972889747
    @test derivative(logerfcx, -2) ≈ -4.010357718006968536874998078
end
