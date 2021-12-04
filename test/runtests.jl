using SafeTestsets, Random, Test

@time @safetestset "util" begin include("util.jl") end
@time @safetestset "onehot" begin include("onehot.jl") end
@time @safetestset "minibatches" begin include("minibatches.jl") end
@time @safetestset "truncnorm" begin include("truncnorm/truncnorm.jl") end
@time @safetestset "truncnorm rejection" begin include("truncnorm/rejection.jl") end
@time @safetestset "layers" begin include("layers.jl") end
@time @safetestset "rbm" begin include("rbm.jl") end

#@time @safetestset "regularize" begin include("regularize.jl") end
# @time @safetestset "pseudolikelihood" begin include("pseudolikelihood.jl") end
# @time @safetestset "learning rate decay schedules" begin include("lr_schedules.jl") end
# @time @safetestset "partition" begin include("partition.jl") end
