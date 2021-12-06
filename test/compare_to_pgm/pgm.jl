using DelimitedFiles
import RestrictedBoltzmannMachines as RBMs

#= Comparisons against https://github.com/jertubiana/PGM.
I computed the energy, free_energy, and so on, using the PGM
code. So here we compare if using the same RBM I obtain the same
values of energies, etc. with my code. =#

@testset "Binary / Binary RBM" begin
    g = vec(readdlm("compare_to_pgm/RBM_bin_bin_gv.txt"));
    θ = vec(readdlm("compare_to_pgm/RBM_bin_bin_gh.txt"));
    w = readdlm("compare_to_pgm/RBM_bin_bin_w.txt")';

    v = readdlm("compare_to_pgm/RBM_bin_v.txt")'
    h = readdlm("compare_to_pgm/RBM_bin_h.txt")'
    E = vec(readdlm("compare_to_pgm/RBM_bin_bin_E.txt"))
    F = vec(readdlm("compare_to_pgm/RBM_bin_bin_F.txt"))

    rbm = RBMs.RBM(RBMs.Binary(g), RBMs.Binary(θ), w)
    @test RBMs.energy(rbm, v, h) ≈ E
end
