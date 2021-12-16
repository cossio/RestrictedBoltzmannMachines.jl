using SafeTestsets, Random, Test

@time @safetestset "util" begin include("util.jl") end
@time @safetestset "linalg" begin include("linalg.jl") end
@time @safetestset "onehot" begin include("onehot.jl") end
@time @safetestset "layers" begin include("layers.jl") end
@time @safetestset "rbm" begin include("rbm.jl") end
@time @safetestset "minibatches" begin include("minibatches.jl") end
@time @safetestset "zerosum" begin include("zerosum.jl") end
@time @safetestset "initialization" begin include("initialization.jl") end
@time @safetestset "pseudolikelihood" begin include("pseudolikelihood.jl") end
@time @safetestset "pgm" begin include("compare_to_pgm/pgm.jl") end
@time @safetestset "partition" begin include("partition.jl") end
@time @safetestset "regularize" begin include("regularize.jl") end
@time @safetestset "truncnorm" begin include("truncnorm.jl") end

# @time @safetestset "learning rate decay schedules" begin include("lr_schedules.jl") end
