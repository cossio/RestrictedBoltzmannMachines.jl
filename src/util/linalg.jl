@doc raw"""
    block_matrix_logdet(A, B, C, D)

Log-determinant of a block matrix using the determinant lemma.

```math
\det\left(
    \begin{bmatrix}
        \mathbf{A} & \mathbf{B} \\
        \mathbf{C} & \mathbf{D}
    \end{bmatrix}
\right)
= \det(A) \det(D - CA^{-1}B)
= \det(D) \det(A - BD^{-1}C)
```

Here we assume that `A` and `D` are invertible, and moreover are easy to invert
(for example, if they are diagonal).
We use this to chose one or the other of the two formulas above.
"""
function block_matrix_logdet(
    A::AbstractMatrix, B::AbstractMatrix,
    C::AbstractMatrix, D::AbstractMatrix
)
    @assert size(A, 1) == size(B, 1)
    @assert size(C, 1) == size(D, 1)
    @assert size(A, 2) == size(C, 2)
    @assert size(B, 2) == size(D, 2)

    if length(A) ≥ length(D)
        return logdet(A) + logdet(D - C * inv(A) * B)
    else
        return logdet(D) + logdet(A - B * inv(D) * C)
    end
end

@doc raw"""
    block_matrix_invert(A, B, C, D)

Inversion of a block matrix, using the formula:

```math
\begin{bmatrix}
    \mathbf{A} & \mathbf{B} \\
    \mathbf{C} & \mathbf{D}
\end{bmatrix}^{-1}
=
\begin{bmatrix}
    \left(\mathbf{A} - \mathbf{B} \mathbf{D}^{-1} \mathbf{C}\right)^{-1} & \mathbf{0} \\
    \mathbf{0} & \left(\mathbf{D} - \mathbf{C} \mathbf{A}^{-1} \mathbf{B}\right)^{-1}
\end{bmatrix}
\begin{bmatrix}
    \mathbf{I} & -\mathbf{B} \mathbf{D}^{-1} \\
    -\mathbf{C} \mathbf{A}^{-1} & \mathbf{I}
\end{bmatrix}
```

Assumes that `A` and `D` are square and invertible.
"""
function block_matrix_invert(
    A::AbstractMatrix, B::AbstractMatrix,
    C::AbstractMatrix, D::AbstractMatrix
)
    @assert size(A, 1) == size(A, 2) == size(B, 1) == size(C, 2)
    @assert size(D, 1) == size(D, 2) == size(B, 2) == size(C, 1)

    CinvA = C * inv(A)
    BinvD = B * inv(D)

    M = [
        inv(A - BinvD * C)  zero(B)
        zero(C)  inv(D - CinvA * B)
    ]

    N = [
        I(size(B, 1))  -BinvD
        -CinvA  I(size(C, 1))
    ]

    return M * N
end
