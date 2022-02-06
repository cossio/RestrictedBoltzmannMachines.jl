using Test: @test, @testset
import DelimitedFiles
import Statistics
import NPZ
import RestrictedBoltzmannMachines as RBMs

#=
Comparisons against https://github.com/jertubiana/PGM.

I computed the energy, free_energy, and so on, using the PGM
code. So here we compare if using the same RBM I obtain the same
values of energies, etc. with my code.

The only difference is in the definition of some θ's (for Gaussian, ReLU, and dReLU layers).
I wanted to be consistent with the definition of the external fields, which appear with
a minus sign in the energy for discrete units. Therefore in the following tests pay
attention to these minus signs.
=#

@testset "Binary / Binary RBM" begin
    g = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_bin_g_v.txt"))
    θ = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_bin_g_h.txt"))
    w = DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_bin_w.txt")'

    v = DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_bin_v.txt")'
    h = DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_bin_h.txt")'
    E = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_bin_E.txt"))
    F = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_bin_F.txt"))

    rbm = RBMs.RBM(RBMs.Binary(g), RBMs.Binary(θ), w)
    @test RBMs.energy(rbm, v, h) ≈ E rtol=1e-5
    @test RBMs.free_energy(rbm, v) ≈ F rtol=1e-5
end

@testset "Binary / Gaussian RBM" begin
    #= Note the minus sign in front of θ =#
    θ = -vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_gauss_theta_h.txt"))
    γ =  vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_gauss_gamma_h.txt"))
    g = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_gauss_g_v.txt"))
    w = DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_gauss_w.txt")'

    v = DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_gauss_v.txt")'
    h = DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_gauss_h.txt")'
    Ev = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_gauss_Ev.txt"))
    Eh = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_gauss_Eh.txt"))
    E = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_gauss_E.txt"))
    F = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_gauss_F.txt"))

    rbm = RBMs.RBM(RBMs.Binary(g), RBMs.Gaussian(θ, γ), w)
    @test RBMs.energy(rbm.visible, v) ≈ Ev rtol=1e-5
    @test RBMs.energy(rbm.hidden,  h) ≈ Eh rtol=1e-5
    @test RBMs.energy(rbm, v, h) ≈ E rtol=1e-5
    @test RBMs.free_energy(rbm, v) ≈ F rtol=1e-5
end

@testset "Binary / dReLU RBM" begin
    #= Note the minus sign in front of θp. For Gaussian, ReLU, and dReLU layers
    (as well as for the reparameterized pReLU), my θ's are minus those of Jerome.
    I did this because I wanted to be consistent with the external field of Binary,
    Spin and Potts layer, for which the field appears with this sign. =#
    θp = -vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_thetap_h.txt"))
    θn =  vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_thetam_h.txt"))
    γp = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_gammap_h.txt"))
    γn = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_gammam_h.txt"))
    g = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_g_v.txt"))
    w = DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_w.txt")'

    v = DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_v.txt")'
    h = DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_h.txt")'
    Ev = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_Ev.txt"))
    Eh = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_Eh.txt"))
    E = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_E.txt"))
    F = vec(DelimitedFiles.readdlm("compare_to_pgm/RBM_bin_dReLU_F.txt"))

    rbm = RBMs.RBM(RBMs.Binary(g), RBMs.dReLU(θp, θn, γp, γn), w)
    @test RBMs.energy(rbm.visible, v) ≈ Ev rtol=1e-5
    @test RBMs.energy(rbm.hidden,  h) ≈ Eh rtol=1e-5
    @test RBMs.energy(rbm, v, h) ≈ E rtol=1e-5
    @test RBMs.free_energy(rbm, v) ≈ F rtol=1e-3
end

@testset "Binary pseudolikelihood" begin
    w = Matrix(DelimitedFiles.readdlm("compare_to_pgm/PL/RBM_Bernoulli_weights.txt")')
    visible_g = vec(DelimitedFiles.readdlm("compare_to_pgm/PL/RBM_Bernoulli_visible_fields.txt"))
    hidden_g = vec(DelimitedFiles.readdlm("compare_to_pgm/PL/RBM_Bernoulli_hidden_fields.txt"))
    v = BitMatrix(DelimitedFiles.readdlm("compare_to_pgm/PL/RBM_Bernoulli_data.txt", Bool)')
    pl = vec(DelimitedFiles.readdlm("compare_to_pgm/PL/RBM_Bernoulli_PL.txt"))

    rbm = RBMs.RBM(RBMs.Binary(visible_g), RBMs.Binary(hidden_g), w)
    pl_rbm = RBMs.log_pseudolikelihood(rbm, v)

    @test Statistics.mean(pl_rbm) ≈ Statistics.mean(pl) rtol=0.05
    @test Statistics.std(pl_rbm)  ≈ Statistics.std(pl)  rtol=0.1
end

@testset "Potts pseudolikelihood" begin
    # https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html
    d = NPZ.npzread("compare_to_pgm/PL/RBM_Potts_PL.npz")
    rbm = RBMs.RBM(
        RBMs.Potts(Matrix(d["g"]')),
        RBMs.dReLU(-d["theta_plus"], d["theta_minus"], d["gamma_plus"], d["gamma_minus"]),
        permutedims(d["weights"], (3,2,1))
    )
    data = RBMs.onehot_encode(d["data"]')
    pl = RBMs.log_pseudolikelihood(rbm, data)

    # TODO: #7 see if we can decrease rtol
    @test Statistics.mean(pl) ≈ Statistics.mean(d["PL"]) rtol=0.1
    @test Statistics.std(pl) ≈ Statistics.std(d["PL"]) rtol=0.1
end
