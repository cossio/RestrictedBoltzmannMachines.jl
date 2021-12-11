import RestrictedBoltzmannMachines as RBMs # make sure the rules we defined are loaded
import ForwardDiff, FiniteDifferences, SpecialFunctions

@testset "∂ logerfcx" begin
    @test ForwardDiff.derivative(SpecialFunctions.logerfcx,  1) ≈ -0.638967514234791260471501152
    @test ForwardDiff.derivative(SpecialFunctions.logerfcx, -1) ≈ -2.225271242628657455193006951
    @test ForwardDiff.derivative(SpecialFunctions.logerfcx,  2) ≈ -0.418160805994414197972889747
    @test ForwardDiff.derivative(SpecialFunctions.logerfcx, -2) ≈ -4.010357718006968536874998078

    fdm = FiniteDifferences.central_fdm(5, 1)
    for x in -10:0.1:10
        ∂f = ForwardDiff.derivative(SpecialFunctions.logerfcx, x)
        Δf = fdm(SpecialFunctions.logerfcx, x)
        @test ∂f ≈ Δf rtol=1e-10
    end
end
