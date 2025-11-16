module HDF5Ext

import HDF5
import RestrictedBoltzmannMachines
using HDF5: h5open
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: Gaussian
using RestrictedBoltzmannMachines: Potts
using RestrictedBoltzmannMachines: PottsGumbel
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: Spin
using RestrictedBoltzmannMachines: StandardizedRBM
using RestrictedBoltzmannMachines: xReLU
using RestrictedBoltzmannMachines: nsReLU

# Version of the file format used to save/load RBMs
const FILE_FORMAT_VERSION = v"1.0.0"

# Header used to identify the file format in the HDF5 file structure
const FILE_FORMAT_HEADER = "rbm_hdf5_file_format_version"

"""
    load_rbm(path)

Load an RBM from an HDF5 file at `path`.
"""
RestrictedBoltzmannMachines.load_rbm(path::AbstractString) = h5open(path, "r") do file
    format_version = read(file, FILE_FORMAT_HEADER)
    if format_version == string(FILE_FORMAT_VERSION)
        rbm_type = read(file, "rbm_type")
        return _load_rbm(file, Val(Symbol(rbm_type)))
    else
        error("Unsupported format version: $format_version")
    end
end

"""
    save_rbm(path, rbm; overwrite=false)

Save an RBM to an HDF5 file at `path`. If `overwrite` is `false` (the default),
an error is thrown if the file already exists.
"""
function RestrictedBoltzmannMachines.save_rbm(path::AbstractString, rbm::RBM; overwrite::Bool=false)
    if !overwrite && isfile(path)
        error("File already exists: $path")
    end
    h5open(path, "w") do file
        write(file, FILE_FORMAT_HEADER, string(FILE_FORMAT_VERSION))
        write(file, "rbm_type", "RBM")
        write(file, "weights", rbm.w)
        write(file, "visible_par", rbm.visible.par)
        write(file, "hidden_par", rbm.hidden.par)
        write(file, "visible_type", layer_type(rbm.visible))
        write(file, "hidden_type", layer_type(rbm.hidden))
    end
    return path
end

function RestrictedBoltzmannMachines.save_rbm(path::AbstractString, rbm::StandardizedRBM; overwrite::Bool=false)
    if !overwrite && isfile(path)
        error("File already exists: $path")
    end
    h5open(path, "w") do file
        write(file, FILE_FORMAT_HEADER, string(FILE_FORMAT_VERSION))
        write(file, "rbm_type", "StandardizedRBM")
        write(file, "weights", rbm.w)
        write(file, "visible_par", rbm.visible.par)
        write(file, "hidden_par", rbm.hidden.par)
        write(file, "visible_type", layer_type(rbm.visible))
        write(file, "hidden_type", layer_type(rbm.hidden))
        write(file, "offset_v", rbm.offset_v)
        write(file, "offset_h", rbm.offset_h)
        write(file, "scale_v", rbm.scale_v)
        write(file, "scale_h", rbm.scale_h)
    end
    return path
end

layer_type(::Binary) = "Binary"
layer_type(::Spin) = "Spin"
layer_type(::Potts) = "Potts"
layer_type(::Gaussian) = "Gaussian"
layer_type(::xReLU) = "xReLU"
layer_type(::nsReLU) = "nsReLU"
layer_type(::PottsGumbel) = "PottsGumbel"

construct_layer(layer_type::AbstractString, par::AbstractArray) = construct_layer(Val(Symbol(layer_type)), par)

construct_layer(::Val{:Binary}, par::AbstractArray) = Binary(par)
construct_layer(::Val{:Spin}, par::AbstractArray) = Spin(par)
construct_layer(::Val{:Potts}, par::AbstractArray) = Potts(par)
construct_layer(::Val{:Gaussian}, par::AbstractArray) = Gaussian(par)
construct_layer(::Val{:xReLU}, par::AbstractArray) = xReLU(par)
construct_layer(::Val{:nsReLU}, par::AbstractArray) = nsReLU(par)
construct_layer(::Val{:PottsGumbel}, par::AbstractArray) = PottsGumbel(par)

function _load_rbm(file::HDF5.File, ::Val{:RBM})
    w = read(file, "weights")
    visible = construct_layer(read(file, "visible_type"), read(file, "visible_par"))
    hidden = construct_layer(read(file, "hidden_type"), read(file, "hidden_par"))
    return RBM(visible, hidden, w)
end

function _load_rbm(file::HDF5.File, ::Val{:StandardizedRBM})
    offset_v = read(file, "offset_v")
    offset_h = read(file, "offset_h")
    scale_v = read(file, "scale_v")
    scale_h = read(file, "scale_h")
    w = read(file, "weights")
    visible = construct_layer(read(file, "visible_type"), read(file, "visible_par"))
    hidden = construct_layer(read(file, "hidden_type"), read(file, "hidden_par"))
    return StandardizedRBM(visible, hidden, w, offset_v, offset_h, scale_v, scale_h)
end

end
