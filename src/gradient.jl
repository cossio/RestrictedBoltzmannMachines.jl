"""
    ∂energy(layer; stats...)

Derivative of average energy of configurations with respect to layer parameters,
where `stats...` refers to the sufficient statistics from samples required by the layer.
See [`sufficient_statistics`](@ref).
"""
function ∂energy end

"""
    ∂free_energy(layer, inputs = 0; wts = 1)

Unit activation moments, conjugate to layer parameters.
These are obtained by differentiating `free_energies` with respect to the layer parameters.
Averages over configurations (weigthed by `wts`).
"""
function ∂free_energy(layer::AbstractLayer, inputs::AbstractArray; wts = nothing)
    @assert size(layer) == size(inputs)[1:ndims(layer)]
    layer_eff = effective(layer, inputs)
    ∂Feff = ∂free_energy(layer_eff)
    if ndims(layer) == ndims(inputs)
        @assert isnothing(wts)
        @assert size(layer_eff) == size(layer)
        return ∂Feff
    else
        return map(∂Feff) do ∂fs
            @assert size(∂fs) == size(layer_eff)
            ∂ω = batchmean(layer, ∂fs; wts)
            @assert size(∂ω) == size(layer)
            ∂ω
        end
    end
end

function ∂free_energy(layer::AbstractLayer, input::Real; wts::Nothing = nothing)
    inputs = FillArrays.Fill(input, size(layer))
    return ∂free_energy(layer, inputs; wts)
end

function ∂free_energy(
    rbm::RBM, v::AbstractArray; wts = nothing,
    stats = sufficient_statistics(rbm.visible, v; wts)
)
    inputs = inputs_v_to_h(rbm, v)
    h = transfer_mean(rbm.hidden, inputs)
    ∂v = ∂energy(rbm.visible; stats...)
    ∂h = ∂free_energy(rbm.hidden, inputs; wts)

    hmat = reshape(h, length(rbm.hidden), :)
    vmat = activations_convert_maybe(hmat, reshape(v, length(rbm.visible), :))
    @assert size(vmat, 2) == size(hmat, 2)
    if isnothing(wts)
        ∂wflat = -vmat * hmat' / size(vmat, 2)
    else
        @assert size(wts) == batchsize(rbm.visible, v)
        @assert size(vmat, 2) == size(hmat, 2) == length(wts)
        ∂wflat = -vmat * Diagonal(vec(wts)) * hmat' / size(vmat, 2)
    end
    @assert size(∂wflat) == (length(rbm.visible), length(rbm.hidden))
    ∂w = reshape(∂wflat, size(rbm.w))

    return (visible = ∂v, hidden = ∂h, w = reshape(∂w, size(rbm.w)))
end
