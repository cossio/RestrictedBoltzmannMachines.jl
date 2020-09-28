export tensormul_ff, tensormul_ll, tensormul_lf, tensormul_fl, tensordot

NumArray{T<:Number,N} = AbstractArray{<:T,N}
Num{T<:Number,N} = Union{T, AbstractArray{<:T,N}}

"""
    tensormul_ff(A, B, Val(dims))

`A*B` contracting first `dims` dimensions of `A` with first `dims` dimensions of `B`.
"""
function tensormul_ff(A::NumArray, B::NumArray, ::Val{dims}) where {dims}
	dims::Int
	@assert ndims(A) ≥ dims && ndims(B) ≥ dims
	A_contracted = ntuple(d -> size(A, d), Val(dims))
	B_contracted = ntuple(d -> size(B, d), Val(dims))
	A_non_contracted = ntuple(d -> size(A, dims + d), Val(ndims(A) - dims))
	B_non_contracted = ntuple(d -> size(B, dims + d), Val(ndims(B) - dims))
	@assert A_contracted == B_contracted
	@assert size(A) == (A_contracted..., A_non_contracted...)
	@assert size(B) == (B_contracted..., B_non_contracted...)
	Amat = reshape(A, :, prod(A_non_contracted))
	Bmat = reshape(B, :, prod(B_non_contracted))
	Cmat = Amat' * Bmat
	C_size = (A_non_contracted..., B_non_contracted...)
	C = reshape(Cmat, C_size)
	return C
end

"""
    tensormul_ll(A, B, Val(dims))

`A*B` contracting last `dims` dimensions of `A` with last `dims` dimensions of `B`.
"""
function tensormul_ll(A::NumArray, B::NumArray, ::Val{dims}) where {dims}
	dims::Int
	@assert ndims(A) ≥ dims && ndims(B) ≥ dims
	A_contracted = ntuple(d -> size(A, ndims(A) - dims + d), Val(dims))
	B_contracted = ntuple(d -> size(B, ndims(B) - dims + d), Val(dims))
	A_non_contracted = ntuple(d -> size(A, d), Val(ndims(A) - dims))
	B_non_contracted = ntuple(d -> size(B, d), Val(ndims(B) - dims))
	@assert A_contracted == B_contracted
	@assert size(A) == (A_non_contracted..., A_contracted...)
	@assert size(B) == (B_non_contracted..., B_contracted...)
	Amat = reshape(A, prod(A_non_contracted), :)
	Bmat = reshape(B, prod(B_non_contracted), :)
	Cmat = Amat * Bmat'
	C_size = (A_non_contracted..., B_non_contracted...)
	C = reshape(Cmat, C_size)
	return C
end

"""
    tensormul_lf(A, B, Val(dims))

`A*B` contracting last `dims` dimensions of `A` with first `dims` dimensions of `B`.
"""
function tensormul_lf(A::NumArray, B::NumArray, ::Val{dims}) where {dims}
	dims::Int
	@assert ndims(A) ≥ dims && ndims(B) ≥ dims
	A_contracted = ntuple(d -> size(A, ndims(A) - dims + d), Val(dims))
	B_contracted = ntuple(d -> size(B, d), Val(dims))
	A_non_contracted = ntuple(d -> size(A, d), ndims(A) - dims)
	B_non_contracted = ntuple(d -> size(B, dims + d), ndims(B) - dims)
	@assert A_contracted == B_contracted
	@assert size(A) == (A_non_contracted..., A_contracted...)
	@assert size(B) == (B_contracted..., B_non_contracted...)
	Amat = reshape(A, prod(A_non_contracted), :)
	Bmat = reshape(B, :, prod(B_non_contracted))
	Cmat = Amat * Bmat
	C_size = (A_non_contracted..., B_non_contracted...)
	C = reshape(Cmat, C_size)
	return C
end

"""
    tensormul_fl(A, B, Val(dims))

`A*B` contracting first `dims` dimensions of `A` with last `dims` dimensions of `B`.
"""
function tensormul_fl(A::NumArray, B::NumArray, ::Val{dims}) where {dims}
	dims::Int
	@assert ndims(A) ≥ dims && ndims(B) ≥ dims
	A_contracted = ntuple(d -> size(A, d), Val(dims))
	B_contracted = ntuple(d -> size(B, ndims(B) - dims + d), Val(dims))
	A_non_contracted = ntuple(d -> size(A, dims + d), ndims(A) - dims)
	B_non_contracted = ntuple(d -> size(B, d), ndims(B) - dims)
	@assert A_contracted == B_contracted
	@assert size(A) == (A_contracted..., A_non_contracted...)
	@assert size(B) == (B_non_contracted..., B_contracted...)
	Amat = reshape(A, :, prod(A_non_contracted))
	Bmat = reshape(B, prod(B_non_contracted), :)
	Cmat = (Bmat * Amat)'
	C_size = (A_non_contracted..., B_non_contracted...)
	return reshape(Cmat, C_size)
end

"""
	tensordot(X, W, Y)

`X*W*Y`, contracting all dimensions of `W` with the corresponding first
dimensions of `X` and `Y`, and matching the remaining last dimensions of
`X` to the remaining last dimensions of `Y`.

For example, `C[b] = sum(X[i,j,b] * W[i,j,μ,ν] * Y[μ,ν,b])`.
"""
function tensordot(X::NumArray, W::NumArray, Y::NumArray)
	xsize, ysize, bsize = tensorsizes(X, W, Y)
	Xmat = reshape(X, prod(xsize), prod(bsize))
	Ymat = reshape(Y, prod(ysize), prod(bsize))
	Wmat = reshape(W, prod(xsize), prod(ysize))
	if size(Wmat, 1) ≥ size(Wmat, 2)
		Cmat = sum(Ymat .* (Wmat' * Xmat); dims=1)
	else
		Cmat = sum(Xmat .* (Wmat * Ymat); dims=1)
	end
	return reshape(Cmat, bsize)
end

function tensorsizes(X::NumArray, W::NumArray, Y::NumArray)
	@assert iseven(ndims(X) + ndims(Y) - ndims(W))
	bdims = div(ndims(X) + ndims(Y) - ndims(W), 2)
	@assert ndims(X) ≥ bdims && ndims(Y) ≥ bdims
	xdims = ndims(X) - bdims
	ydims = ndims(Y) - bdims
	xsize = ntuple(d -> size(X, d), xdims)
	ysize = ntuple(d -> size(Y, d), ydims)
	bsize = ntuple(d -> size(X, d + xdims), bdims)
	@assert size(W) == (xsize..., ysize...)
	@assert size(X) == (xsize..., bsize...)
	@assert size(Y) == (ysize..., bsize...)
	return xsize, ysize, bsize
end

"""
    broadlike(A, B...)

Broadcasts `A` to the size of `A .+ B .+ ...`, without actually summing anything.
"""
broadlike(A, B...) = broadcast(first ∘ tuple, A, B...)
