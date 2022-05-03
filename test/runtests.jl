#= Github Actions uses Intel CPUs, so it is faster to use MKL than OpenBLAS.
It is recommended to load MKL before ANY other package.=#
#= Wait until https://github.com/JuliaLinearAlgebra/MKL.jl/issues/112 is resolved
before using MKL on CI for Mac =#
#import MKL
using SafeTestsets: @safetestset

@time @safetestset "util" begin include("util.jl") end
@time @safetestset "linalg" begin include("linalg.jl") end
@time @safetestset "onehot" begin include("onehot.jl") end
@time @safetestset "rbm" begin include("rbm.jl") end
@time @safetestset "layers" begin include("layers.jl") end
@time @safetestset "cd" begin include("cd.jl") end
@time @safetestset "pseudolikelihood" begin include("pseudolikelihood.jl") end
@time @safetestset "minibatches" begin include("minibatches.jl") end
@time @safetestset "initialization" begin include("initialization.jl") end
@time @safetestset "regularize" begin include("regularize.jl") end
@time @safetestset "truncnorm" begin include("truncnorm.jl") end
@time @safetestset "optim" begin include("optim.jl") end
@time @safetestset "partition" begin include("partition.jl") end
@time @safetestset "ais" begin include("ais.jl") end
@time @safetestset "metropolis" begin include("metropolis.jl") end

@time @safetestset "zerosum" begin include("gauge/zerosum.jl") end
@time @safetestset "rescale_hidden" begin include("gauge/rescale_hidden.jl") end
@time @safetestset "centered_gradient" begin include("centered_gradient.jl") end
@time @safetestset "pcd" begin include("pcd.jl") end

if Sys.islinux() # NPZ has issues on Windows
    # compare to https://github.com/jertubiana/PGM
    @time @safetestset "pgm" begin include("compare_to_pgm/pgm.jl") end
end
