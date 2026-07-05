# Design and performance notes

Records of design decisions that were benchmarked, so they are not revisited
without rechecking the numbers. Benchmarks below were run in July 2026 on an
NVIDIA RTX PRO 6000 (GPU) and a multithreaded CPU with OpenBLAS, with
`Float32` weights and typical protein-model sizes (`Q ≈ 5–21`,
visible `≈ 2_000–10_000`, hidden `≈ 100–1_000`, batch `≈ 256–4_000`).

## Why Potts samples are onehot `BitArray`s, not floats

`sample_from_inputs` for `Potts` and `PottsGumbel` returns a onehot
`BitArray`. The float conversion happens later, inside `inputs_h_from_v` /
`inputs_v_from_h`, where `with_eltype_of` converts the sample to the weight
eltype right before the matrix multiplication.

Returning floats from the sampler (to skip that conversion) is not worth it:

- **GPU:** the conversion is within noise next to the `w'v` matmul
  (measured speedups of 0.84–1.17× across sizes, i.e. none), while float
  onehots occupy 32× the memory of a `BitArray`. Persistent chains in PCD
  would pay that memory cost permanently, for no speed gain.
- **CPU:** the `BitArray → Float32` unpack is 27–57% of the input projection,
  so dropping it looks like a 1.9–2.9× win on that step in isolation. But
  part of that cost would simply move into the sampler, which would then
  write a 32× larger float array instead of a compact `BitArray`.

## Why onehot × weights is a dense matmul, not indexing / gather

Multiplying a onehot array by the weight matrix is mathematically a row
selection, and libraries with huge vocabularies implement it that way:
Flux.jl's `OneHotArray` overloads `*` as indexing (`NNlib.gather`), and
PyTorch's `nn.Embedding` is an `index_select`. That pays off when the
category dimension is large (vocabulary sizes of ``10^4``–``10^5``): the
dense product would waste time multiplying by zeros.

Here `Q` is small (2–21), so the "wasted" factor is small and the dense
product runs at gemm speed. Benchmarks: CUBLAS beats an `NNlib.gather`-based
path by 4–8× on GPU; on CPU, gather only wins (~1.5×) for very large models.
A gather path also needs a custom kernel to avoid materializing a
`hidden × N × batch` intermediate when summing over sites. Not worth the
complexity at these sizes.

## Categorical sampling: inverse-CDF vs the Gumbel trick

`Potts` samples with `categorical_sample_from_logits`, an inverse-CDF scalar
loop (softmax, one uniform per site, linear scan). This is the fastest option
on CPU (~2.5–3× faster than Gumbel-argmax) but cannot run on GPU (scalar
indexing).

`PottsGumbel` samples with the Gumbel-argmax trick
(`argmax(logits .+ gumbel_noise)`), which is GPU-friendly but draws `Q`
random numbers per site instead of one.

For reference, PyTorch's `Categorical.sample()` on GPU does *not* use Gumbel:
`torch.multinomial` with one sample uses a fused inverse-CDF kernel
(`sampleMultinomialOnce`: per-distribution prefix sum + one uniform + bucket
search). PyTorch reserves Gumbel for the differentiable relaxation
(`gumbel_softmax`) and for without-replacement sampling — neither of which
Gibbs sampling needs.

A fused inverse-CDF CUDA kernel (one thread per site: max, exp-sum, one
uniform, linear scan over `Q`) measured **4.6–16× faster** than
Gumbel-argmax on GPU, because it draws `Q`× fewer random numbers and
materializes no `Q × N × B` temporaries. Note the fusion is essential: a
*vectorized* inverse-CDF built from library ops (`softmax → cumsum → count`)
is **slower** than Gumbel (0.4–0.77×) because of the intermediates. If GPU
Potts sampling ever becomes a bottleneck, the right move is such a kernel in
the CUDA extension — it is exact (no relaxation) and would make `PottsGumbel`
unnecessary for speed.
