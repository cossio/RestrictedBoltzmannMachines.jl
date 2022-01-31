using Test, Random, LinearAlgebra, Statistics, DelimitedFiles
import RestrictedBoltzmannMachines as RBMs

@testset "subtract_gradients" begin
    nt1 = (x = [2], y = [3])
    nt2 = (x = [1], y = [-1])
    @test RBMs.subtract_gradients(nt1, nt2) == (x = [1], y = [4])

    nt1 = (x = [2], y = [3], t = (a = [1], b = [2]))
    nt2 = (x = [1], y = [-1], t = (a = [2], b = [0]))
    @test RBMs.subtract_gradients(nt1, nt2) == (
        x = [1], y = [4], t = (a = [-1], b = [2])
    )
end
