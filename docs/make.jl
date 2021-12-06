using Documenter, Literate
import RestrictedBoltzmannMachines as RBMs

examples = Vector{Pair{String,String}}()
for ex in readdir(joinpath(@__DIR__, "literate"); join=true)
    if endswith(ex, ".jl")
        name = first(splitext(basename(ex)))
        Literate.markdown(ex, joinpath(@__DIR__, "src/literate"); name=name)
        push!(examples, name => "literate/$name.md")
    end
end

makedocs(
    modules=[RBMs],
    sitename="RestrictedBoltzmannMachines.jl",
    pages = [
        "Examples" => examples
    ]
)

deploydocs(
    repo = "github.com/cossio/RestrictedBoltzmannMachines.jl.git",
    devbranch = "master"
)
