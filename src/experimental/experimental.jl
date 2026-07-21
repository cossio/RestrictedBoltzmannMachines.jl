"""
    Experimental

Namespace for experimental features. Everything under `Experimental` — including the
[`Experimental.Sparse`](@ref) submodule — is unstable and may change arbitrarily at any
time, without a breaking release. Do not depend on it in code that needs a stable API.
"""
module Experimental

# NOTE: this module and everything nested under it is experimental. Its contents (names,
# signatures, behavior) may change arbitrarily at any time, without a breaking release.

include("sparse.jl")

end # module Experimental
