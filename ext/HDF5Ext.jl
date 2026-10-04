module HDF5Ext

import HDF5
import RestrictedBoltzmannMachines
using HDF5: h5open
using RestrictedBoltzmannMachines: RBM, CenteredRBM, StandardizedRBM
using RestrictedBoltzmannMachines: Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU,
    dReLU, pReLU, xReLU, nsReLU

# Version of the file format used to save/load RBMs
const FILE_FORMAT_VERSION = v"1.0.0"

# Header used to identify the file format in the HDF5 file structure
const FILE_FORMAT_HEADER = "rbm_hdf5_file_format_version"

RestrictedBoltzmannMachines.load_rbm(path::AbstractString) = h5open(path, "r") do file
    format_version = read(file, FILE_FORMAT_HEADER)
    format_version == string(FILE_FORMAT_VERSION) ||
        error("Unsupported format version: $format_version")
    T = construct_rbm(read(file, "rbm_type"))
    visible = construct_layer(read(file, "visible_type"), read(file, "visible_par"))
    hidden = construct_layer(read(file, "hidden_type"), read(file, "hidden_par"))
    extras = (read(file, string(f)) for f in extra_fields(T))
    return T(visible, hidden, read(file, "weights"), extras...)
end

function RestrictedBoltzmannMachines.save_rbm(
        path::AbstractString, rbm::Union{RBM, StandardizedRBM}; overwrite::Bool = false
    )
    !overwrite && isfile(path) && error("File already exists: $path")
    h5open(path, "w") do file
        write(file, FILE_FORMAT_HEADER, string(FILE_FORMAT_VERSION))
        write(file, "rbm_type", rbm_type(rbm))
        write(file, "weights", rbm.w)
        write(file, "visible_par", rbm.visible.par)
        write(file, "hidden_par", rbm.hidden.par)
        write(file, "visible_type", layer_type(rbm.visible))
        write(file, "hidden_type", layer_type(rbm.hidden))
        for f in extra_fields(typeof(rbm))
            write(file, string(f), getfield(rbm, f))
        end
    end
    return path
end

# fields saved besides the layers and weights, in constructor order; a `CenteredRBM` (a
# `StandardizedRBM` with unit scales) is saved without its scales
extra_fields(::Type{<:RBM}) = ()
extra_fields(::Type{<:CenteredRBM}) = (:offset_v, :offset_h)
extra_fields(::Type{<:StandardizedRBM}) = (:offset_v, :offset_h, :scale_v, :scale_h)

# The type names stored in the file are an explicit allow-list: only these can be loaded.
rbm_type(::RBM) = "RBM"
rbm_type(::CenteredRBM) = "CenteredRBM"
rbm_type(::StandardizedRBM) = "StandardizedRBM"

construct_rbm(rbm_type::AbstractString) = construct_rbm(Val(Symbol(rbm_type)))
construct_rbm(::Val{:RBM}) = RBM
construct_rbm(::Val{:CenteredRBM}) = CenteredRBM
construct_rbm(::Val{:StandardizedRBM}) = StandardizedRBM

layer_type(::Binary) = "Binary"
layer_type(::Spin) = "Spin"
layer_type(::Potts) = "Potts"
layer_type(::PottsGumbel) = "PottsGumbel"
layer_type(::Gaussian) = "Gaussian"
layer_type(::ReLU) = "ReLU"
layer_type(::dReLU) = "dReLU"
layer_type(::pReLU) = "pReLU"
layer_type(::xReLU) = "xReLU"
layer_type(::nsReLU) = "nsReLU"

construct_layer(layer_type::AbstractString, par::AbstractArray) = construct_layer(Val(Symbol(layer_type)), par)
construct_layer(::Val{:Binary}, par::AbstractArray) = Binary(par)
construct_layer(::Val{:Spin}, par::AbstractArray) = Spin(par)
construct_layer(::Val{:Potts}, par::AbstractArray) = Potts(par)
construct_layer(::Val{:PottsGumbel}, par::AbstractArray) = PottsGumbel(par)
construct_layer(::Val{:Gaussian}, par::AbstractArray) = Gaussian(par)
construct_layer(::Val{:ReLU}, par::AbstractArray) = ReLU(par)
construct_layer(::Val{:dReLU}, par::AbstractArray) = dReLU(par)
construct_layer(::Val{:pReLU}, par::AbstractArray) = pReLU(par)
construct_layer(::Val{:xReLU}, par::AbstractArray) = xReLU(par)
construct_layer(::Val{:nsReLU}, par::AbstractArray) = nsReLU(par)

end
