# Test files are independent modules. GitHub CI splits them into the groups
# below and runs each group as a separate parallel job. Keep the groups
# roughly balanced in runtime (see timings in .github/workflows/ci.yml runs).
# Set RBM_TEST_GROUP=1..4 to run a single group; leave it unset (or set it to
# "all") to run the full suite, as a plain `Pkg.test()` does.
const TEST_GROUPS = (
    [ # group 1
        "layers.jl",
        "prelu_validation.jl",
        "truncnorm.jl",
        "optim.jl",
        "layers/nsReLU.jl",
    ],
    [ # group 2
        "util.jl",
        "onehot.jl",
        "rbm.jl",
        "shift_fields.jl",
        "standardized.jl",
    ],
    [ # group 3
        "gauge/zerosum.jl",
        "pottsgumbel.jl",
        "centered.jl",
        "jlarrays.jl",
    ],
    [ # group 4
        "pseudolikelihood.jl",
        "infinite_minibatches.jl",
        "initialization.jl",
        "regularize.jl",
        "partition.jl",
        "metropolis.jl",
        "sampling_stationarity.jl",
        "gauge/rescale_hidden.jl",
        "pcd.jl",
        "zero_weight_training.jl",
        "train.jl",
        "explicit_imports.jl",
        "hdf5.jl",
        "ais.jl",
        "aqua.jl",
    ],
)

group = get(ENV, "RBM_TEST_GROUP", "all")
test_files = group == "all" ? reduce(vcat, TEST_GROUPS) : TEST_GROUPS[parse(Int, group)]

for file in test_files
    name = Symbol(replace(file, r"\.jl$" => "", "/" => "_"), :_tests)
    path = joinpath(@__DIR__, file)
    @eval module $name
    include($path)
    end
end
