import Documenter
import Literate
import RestrictedBoltzmannMachines

ENV["JULIA_DEBUG"] = "Documenter,Literate,RestrictedBoltzmannMachines"

# Literate writes a .md next to each .jl under docs/src/literate; clear stale ones before
# and after the build so a renamed source cannot leave an orphan page behind.
const literate_dir = joinpath(@__DIR__, "src/literate")

function clear_md_files(dir::String)
    for (root, dirs, files) in walkdir(dir)
        for file in files
            if endswith(file, ".md")
                rm(joinpath(root, file))
            end
        end
    end
    return
end

clear_md_files(literate_dir)

for (root, dirs, files) in walkdir(literate_dir)
    for file in files
        if endswith(file, ".jl")
            Literate.markdown(joinpath(root, file), root; documenter = true)
        end
    end
end

Documenter.makedocs(
    modules = [RestrictedBoltzmannMachines],
    sitename = "RestrictedBoltzmannMachines.jl",
    pages = [
        "Home" => "index.md",
        "Training" => "training.md",
        "Layer Types" => "layers.md",
        "Examples" => [
            "MNIST" => "literate/MNIST.md",
            "Layers" => [
                "Gaussian" => "literate/layers/Gaussian.md",
                "ReLU" => "literate/layers/ReLU.md",
                "dReLU family" => "literate/layers/dReLU.md",
            ],
            "AIS" => "literate/ais.md",
            "Metropolis" => "literate/metropolis.md",
        ],
        "Developer notes" => [
            "Architecture" => "developer/architecture.md",
            "Design & performance notes" => "developer/design_notes.md",
            "Testing & releasing" => "developer/testing.md",
        ],
        "Reference" => "reference.md",
    ]
)

clear_md_files(literate_dir)

Documenter.deploydocs(
    repo = "github.com/cossio/RestrictedBoltzmannMachines.jl.git",
    devbranch = "master"
)
