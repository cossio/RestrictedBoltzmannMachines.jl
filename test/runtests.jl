using SafeTestsets: @safetestset

module util_tests include("util.jl") end
module linalg_tests include("linalg.jl") end
module onehot_tests include("onehot.jl") end
module rbm_tests include("rbm.jl") end
module layers_tests include("layers.jl") end
module pseudolikelihood_tests include("pseudolikelihood.jl") end
# module minibatches include("minibatches.jl") end
module infinite_minibatches_tests include("infinite_minibatches.jl") end
module initialization_tests include("initialization.jl") end
module regularize_tests include("regularize.jl") end
module truncnorm_tests include("truncnorm.jl") end
module optim_tests include("optim.jl") end
module partition_tests include("partition.jl") end
module metropolis_tests include("metropolis.jl") end

module zerosum_tests include("gauge/zerosum.jl") end
module rescale_hidden_tests include("gauge/rescale_hidden.jl") end
module pcd_tests include("pcd.jl") end

module shift_fields_tests include("shift_fields.jl") end

module aqua_tests include("aqua.jl") end
