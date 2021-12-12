
"""
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
    if length(A) ≥ length(D)
        return logdet(A) + logdet(D - C * inv(A) * B)
    else
        return logdet(D) + logdet(A - B * inv(D) * C)
    end
end

"""
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

Note that this assumes that `A` and `D` are both invertible.
"""
function block_matrix_invert(
    A::AbstractMatrix, B::AbstractMatrix,
    C::AbstractMatrix, D::AbstractMatrix
)
    a = inv(A)
    d = inv(D)

    M = [
        inv(A - B * d * C)  0I
        0I  inv(D - C * a * B)
    ]

    N = [
        I  -B * d
        -C * a  I
    ]

    return M * N
end
