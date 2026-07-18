import CodeComplexity as CC
using Test: @test, @testset

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SOURCE_DIRS = (joinpath(ROOT, "src"), joinpath(ROOT, "ext"))

# These are ratchets for exceptions present on master at 561dd05b, not targets.
# New definitions must stay at or below the general limits, and these
# definitions must exactly match their recorded values. Lower a recorded value
# when an exception is simplified, or remove it once it meets the general
# limit. Lines deliberately identify individual methods rather than granting a
# raised ceiling to every same-named method in a file.
const CYCLOMATIC_EXCEPTIONS = Dict(
    ("src/train/ucd.jl", 36, "_maximal_coupling_step") => 13,
    ("src/train/ucd.jl", 172, "ucd!") => 14,
)
const COGNITIVE_EXCEPTIONS = Dict(
    ("src/train/ucd.jl", 36, "_maximal_coupling_step") => 19,
    ("src/train/ucd.jl", 172, "ucd!") => 16,
)
const ARGUMENT_COUNT_EXCEPTIONS = Dict(
    ("src/centered.jl", 440, "pcd!") => 17,
    ("src/standardized.jl", 430, "pcd!") => 23,
    ("src/train/pcd.jl", 39, "pcd!") => 19,
    ("src/train/ucd.jl", 172, "ucd!") => 22,
)

function source_files()
    files = String[]
    for source_dir in SOURCE_DIRS
        for (dir, _, names) in walkdir(source_dir)
            for name in names
                endswith(name, ".jl") && push!(files, joinpath(dir, name))
            end
        end
    end
    return sort(files)
end

function check_metric(label, metric, limit, exceptions)
    @testset "$label ≤ $limit" begin
        measured_violations = Set{Tuple{String, Int, String}}()
        for filepath in source_files()
            # Call measure_file directly so analyzer errors fail CI instead of
            # being logged and skipped by measure_directory.
            file = CC.measure_file(metric, filepath; max_value = limit)
            path = replace(relpath(filepath, ROOT), '\\' => '/')
            for definition in file.functions
                key = (path, definition.line, definition.name)
                @testset "$path:$(definition.line) $(definition.name)" begin
                    @test key ∉ measured_violations
                    push!(measured_violations, key)
                    @test haskey(exceptions, key)
                    if haskey(exceptions, key)
                        @test definition.value == exceptions[key]
                    end
                end
            end
        end
        @test measured_violations == Set(keys(exceptions))
        @test all(>(limit), values(exceptions))
    end
end

@testset "source and extension complexity ratchets" begin
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
