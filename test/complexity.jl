import CodeComplexity as CC
using Test: @test, @testset

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SOURCE = joinpath(ROOT, "src")

# These are ratchets for exceptions present on master at 561dd05b, not targets.
# New definitions must stay at or below the general limits, and these
# definitions may not get worse. Lower a recorded ceiling when an exception is
# simplified.
const CYCLOMATIC_EXCEPTIONS = Dict(
    ("src/train/ucd.jl", "_maximal_coupling_step") => 13,
    ("src/train/ucd.jl", "ucd!") => 14,
)
const COGNITIVE_EXCEPTIONS = Dict(
    ("src/train/ucd.jl", "_maximal_coupling_step") => 19,
    ("src/train/ucd.jl", "ucd!") => 16,
)
const ARGUMENT_COUNT_EXCEPTIONS = Dict(
    ("src/centered.jl", "pcd!") => 17,
    ("src/standardized.jl", "pcd!") => 23,
    ("src/train/pcd.jl", "pcd!") => 19,
    ("src/train/ucd.jl", "ucd!") => 22,
)

function check_metric(label, metric, limit, exceptions)
    @testset "$label ≤ $limit" begin
        for file in CC.measure_directory(metric, SOURCE; max_value = limit)
            path = replace(relpath(file.path, ROOT), '\\' => '/')
            for definition in file.functions
                key = (path, definition.name)
                ceiling = get(exceptions, key, limit)
                @testset "$path:$(definition.line) $(definition.name)" begin
                    @test definition.value ≤ ceiling
                end
            end
        end
    end
end

@testset "source complexity ratchets" begin
    check_metric(
        "cyclomatic complexity",
        CC.CyclomaticComplexity(),
        10,
        CYCLOMATIC_EXCEPTIONS,
    )
    check_metric(
        "cognitive complexity",
        CC.CognitiveComplexity(),
        15,
        COGNITIVE_EXCEPTIONS,
    )
    check_metric(
        "argument count",
        CC.ArgumentCountComplexity(),
        10,
        ARGUMENT_COUNT_EXCEPTIONS,
    )
end
