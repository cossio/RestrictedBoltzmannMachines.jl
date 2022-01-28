#= As far as I know, Github Actions uses Intel CPUs.
So it is faster to use MKL than OpenBLAS.
It is recommended to load MKL before ANY other package.=#
using MKL, LinearAlgebra

if VERSION ≥ v"1.7"
    @show BLAS.get_config()
else
    @show BLAS.vendor()
end


using SafeTestsets, Random, Test

@time @safetestset "util" begin include("util.jl") end
@time @safetestset "linalg" begin include("linalg.jl") end
@time @safetestset "onehot" begin include("onehot.jl") end
@time @safetestset "rbm" begin include("rbm.jl") end
@time @safetestset "layers" begin include("layers.jl") end
@time @safetestset "cd" begin include("cd.jl") end
@time @safetestset "pseudolikelihood" begin include("pseudolikelihood.jl") end
@time @safetestset "minibatches" begin include("minibatches.jl") end
@time @safetestset "zerosum" begin include("zerosum.jl") end
@time @safetestset "initialization" begin include("initialization.jl") end
@time @safetestset "regularize" begin include("regularize.jl") end
@time @safetestset "truncnorm" begin include("truncnorm.jl") end
@time @safetestset "optim" begin include("optim.jl") end
@time @safetestset "partition" begin include("partition.jl") end

@time @safetestset "centering" begin include("centering.jl") end
@time @safetestset "weight normalization" begin include("wnorm.jl") end

@time @safetestset "pgm" begin include("compare_to_pgm/pgm.jl") end
