#=
Test coverage analysis script.

Usage:
    # Step 1: Run tests with coverage tracking
    julia --project=test --code-coverage=user -e 'include("test/runtests.jl")'

    # Step 2: Print summary report
    julia --project=test test/coverage.jl

    # Step 2 (alternative): Print report with uncovered lines for files below threshold
    julia --project=test test/coverage.jl --uncovered
    julia --project=test test/coverage.jl --uncovered --threshold=80
=#

using Coverage

show_uncovered = "--uncovered" in ARGS
threshold = 90.0
for arg in ARGS
    if startswith(arg, "--threshold=")
        threshold = parse(Float64, split(arg, "=")[2])
    end
end

coverage = process_folder("src")

println("="^80)
println("FILE COVERAGE REPORT")
println("="^80)
println()

global total_hit = 0
global total_miss = 0

results = []
for fc in coverage
    hit = count(x -> x !== nothing && x > 0, fc.coverage)
    miss = count(x -> x !== nothing && x == 0, fc.coverage)
    total = hit + miss
    pct = total > 0 ? round(100 * hit / total; digits = 1) : 100.0
    relpath = replace(fc.filename, pwd() * "/" => "")
    uncovered = [i for (i, c) in enumerate(fc.coverage) if c !== nothing && c == 0]
    push!(results, (relpath, hit, miss, total, pct, uncovered))
    global total_hit += hit
    global total_miss += miss
end

sort!(results, by = x -> x[5])

for (path, hit, miss, total, pct, uncovered) in results
    bar = pct >= 90 ? "✓" : pct >= 70 ? "~" : "✗"
    println("  $bar $(lpad(string(pct), 5))%  $(lpad(string(hit), 4))/$(lpad(string(total), 4))  $path")
    if show_uncovered && pct < threshold && !isempty(uncovered)
        println("      uncovered lines: ", join(uncovered, ", "))
    end
end

println()
total_total = total_hit + total_miss
total_pct = round(100 * total_hit / total_total; digits = 1)
println("TOTAL: $total_pct%  ($total_hit / $total_total lines)")
