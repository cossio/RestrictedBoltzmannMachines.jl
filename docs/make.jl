using Documenter, Literate
import RestrictedBoltzmannMachines as RBMs

examples = Vector{Pair{String,String}}()
for ex in readdir("literate"; join=true)
    if endswith(ex, ".jl")
        name = first(splitext(basename(ex)))
        @show name
        Literate.markdown(ex, "src/literate"; name=name)
        push!(examples, name => "literate/$name.md")
    end
end

@show examples

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
