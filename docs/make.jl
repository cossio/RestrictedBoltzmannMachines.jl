using Documenter, Literate
import RestrictedBoltzmannMachines as RBMs

examples = Vector{Pair{String,String}}()
for ex in readdir("literate"; join=true)
    name = first(splitext(basename(ex)))
    Literate.markdown(ex, "literate/"; name=name)
    push!(examples, name => "literate/$name.md")
end

makedocs(
    modules=[RestrictedBoltzmannMachines],
    sitename="RestrictedBoltzmannMachines.jl",
    pages = [
        "Examples" => examples
    ]
)

deploydocs(
    repo = "github.com/cossio/RestrictedBoltzmannMachines.jl.git",
    devbranch = "master"
)
