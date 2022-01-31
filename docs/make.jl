#= Github Actions uses Intel CPUs, for which MKL is faster than OpenBLAS.
It is recommended to load MKL before any other package, so we load it here. =#
using MKL
using Documenter, Literate
import RestrictedBoltzmannMachines as RBMs

ENV["JULIA_DEBUG"] = "Documenter,Literate,RestrictedBoltzmannMachines"

#=
We place Literate.jl source .jl files and the generated .md files inside docs/src/literate.
=#
const literate_dir = joinpath(@__DIR__, "src/literate")

#=
Helper function to remove all "*.md" files from a directory.
=#
function clear_md_files(dir::String)
    for (root, dirs, files) in walkdir(dir)
        for file in files
            if endswith(file, ".md")
                rm(joinpath(root, file))
            end
        end
    end
end

#=
Remove previously Literate.jl generated files. This removes all "*.md" files inside
`literate_dir`. This is a precaution: if we build docs locally and something fails,
and then change the name of a source file (".jl"), we will be left with a lingering
".md" file which will be included in the current docs build. The following line makes
sure this doesn't happen.
=#
clear_md_files(literate_dir)

#=
Run Literate.jl on the .jl source files within docs/src/literate (recursively).
For each .jl file, this creates a markdown .md file at the same location as
and with the same name as the corresponding .jl file, but with the extension
changed (.jl -> .md).
=#
for (root, dirs, files) in walkdir(literate_dir)
    for file in files
        if endswith(file, ".jl")
            Literate.markdown(joinpath(root, file), root)
        end
    end
end

#=
Build docs.
=#
makedocs(
    modules = [RBMs],
    sitename = "RestrictedBoltzmannMachines.jl",
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "MKL" => "literate/mkl.md",
            "Learning rate decay" => "literate/lr_decay.md",
            "Layers" => [
                "Gaussian" => "literate/layers/Gaussian.md",
                "ReLU" => "literate/layers/ReLU.md",
                "dReLU" => "literate/layers/dReLU.md",
            ],
            "MNIST" => [
                "literate/MNIST/MNIST.md",
                "literate/MNIST/wnorm.md",
                "literate/MNIST/white.md",
            ],
        ],
        "Reference" => "reference.md"
    ],
    strict = true
)

#=
After the docs have been compiled, we can remove the *.md files generated by Literate.
=#
clear_md_files(literate_dir)

#=
Deploy docs to Github pages.
=#
deploydocs(
    repo = "github.com/cossio/RestrictedBoltzmannMachines.jl.git",
    devbranch = "master"
)
