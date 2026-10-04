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
        path::AbstractString, rbm::Union{RBM, StandardizedRBM, CenteredRBM}; overwrite::Bool = false
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

# fields saved besides the layers and weights, in constructor order
extra_fields(::Type{<:RBM}) = ()
extra_fields(::Type{<:CenteredRBM}) = (:offset_v, :offset_h)
extra_fields(::Type{<:StandardizedRBM}) = (:offset_v, :offset_h, :scale_v, :scale_h)

# The type names stored in the file are an explicit allow-list: only these can be loaded.
for T in (RBM, CenteredRBM, StandardizedRBM)
    @eval rbm_type(::$T) = $(string(nameof(T)))
    @eval construct_rbm(::Val{$(QuoteNode(nameof(T)))}) = $T
end
construct_rbm(rbm_type::AbstractString) = construct_rbm(Val(Symbol(rbm_type)))

for T in (Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, dReLU, pReLU, xReLU, nsReLU)
    @eval layer_type(::$T) = $(string(nameof(T)))
    @eval construct_layer(::Val{$(QuoteNode(nameof(T)))}, par::AbstractArray) = $T(par)
end
construct_layer(layer_type::AbstractString, par::AbstractArray) = construct_layer(Val(Symbol(layer_type)), par)

end
