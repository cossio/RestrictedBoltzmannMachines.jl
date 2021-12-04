"""
    save_table(rbm, path)

Saves the parameters of an RBM as a table to a file.
"""
function save_table(rbm::RBM, path::String)
    g = parameter_table(rbm.visible)
    θ = parameter_table(rbm.hidden)
    w = reshape(rbm.weights, length(rbm.visible), length(rbm.hidden))

    p = [w g;
         θ' fill(missing, size(θ, 2), size(g, 2))]

    writedlm(path, p)
end

"""
    load_table!(rbm, path)

Writes the parameters in the file `path` to the `rbm`.
"""
function load_table!(rbm::RBM, path::String)
    p = readdlm(path)
    w = p[1:length(rbm.visible), 1:length(rbm.hidden)]
    g = p[1:length(rbm.visible), (length(rbm.hidden) + 1):end]
    θ = p[(length(rbm.visible) + 1):end, 1:length(rbm.hidden)]
    @assert all(ismissing.(p[(length(rbm.visible) + 1):end, (length(rbm.hidden) + 1):end]))
    rbm.weights .= reshape(w, size(rbm.weights))
    parameter_table!(rbm.visible, g)
    parameter_table!(rbm.hidden, θ)
    return nothing
end

parameter_table(layer::Union{Binary, Spin, Potts}) = [vec(layer.θ)]
parameter_table(layer::Union{Gaussian, ReLU}) = [vec(layer.θ) vec(layer.γ)]
parameter_table(layer::dReLU) = [vec(layer.θp) vec(layer.θn) vec(layer.γp) vec(layer.γn)]

function parameter_table!(layer::Union{Binary, Spins, Potts}, θ::AbstractMatrix)
    @assert size(θ) == (length(layer), 1)
    layer.θ .= reshape(θ, size(layer))
    return nothing
end

function parameter_table!(layer::Union{Gassian, ReLU}, θ::AbstractMatrix)
    @assert size(θ) == (length(layer), 2)
    layer.θ .= reshape(θ[:,1], size(layer))
    layer.γ .= reshape(θ[:,2], size(layer))
    return nothing
end

function parameter_table!(layer::dReLU, θ::AbstractMatrix)
    @assert size(θ) == (length(layer), 4)
    layer.θp .= reshape(θ[:,1], size(layer))
    layer.θm .= reshape(θ[:,2], size(layer))
    layer.γp .= reshape(θ[:,3], size(layer))
    layer.γm .= reshape(θ[:,4], size(layer))
    return nothing
end
