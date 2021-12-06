using Documenter, Literate
import RestrictedBoltzmannMachines as RBMs

#=
We place Literate.jl source .jl files and the generated .md files inside docs/src/literate.
=#
literate_dir = joinpath(@__DIR__, "src/literate")

#=
Clear previous Literate.jl generated files.
This removes all "*.md" files inside `literate_dir`.
=#
for file in readdir(literate_dir; join=true)
    if endswith(file, ".md")
        rm(file)
    end
end

#=
Helper function to extract example names from file paths.
=#
example_name(file) = basename(file)[1:(end - 3)]

#=
Run Literate.jl on the .jl source files within docs/literate.
This creates the markdown .md files inside docs/src/literate.
=#
for file in readdir(literate_dir; join=true)
    if endswith(file, ".jl")
        Literate.markdown(file, literate_dir; name = example_name(file))
    end
end

examples = [
    example_name(file) => file
    for file in readdir(literate_dir; join=true)
    if endswith(file, ".md")
]

makedocs(
    modules = [RBMs],
    sitename = "RestrictedBoltzmannMachines.jl",
    pages = [
        "Home" => "index.md",
        "Mathematical introduction" => "math.md",
        "Examples" => [
            "MNIST" => "literate/MNIST.md"
        ],
        "Reference" => "reference.md"
    ]
)

deploydocs(
    repo = "github.com/cossio/RestrictedBoltzmannMachines.jl.git",
    devbranch = "master"
)
