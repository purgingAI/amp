# How this model is trained and served, term by term

Written to be read once end to end, then returned to. Every number in it was measured on
this machine, and where a well-known technique turned out not to help, that is recorded
too — a list of techniques we "use" that were never measured is worth nothing.

**The single most important fact for reading the rest of this document:** our intent model
is an **encoder**, not a generative LLM. It reads one sentence and emits one label. It does
not produce text token by token. Most of the vLLM-era vocabulary describes problems that
only exist when a model generates a long output one token at a time, and saying we "use"
those techniques would be a claim about a loop we do not run.

---

## Part 1 — the eleven terms

### 1. Prefill

**What it is.** In a generative LLM, answering has two phases. **Prefill** is the first: the
whole prompt is pushed through the network in ONE forward pass, all tokens at once, because
every token of the prompt is already known. It is compute-bound — a big matrix multiply
that saturates the hardware.

**The shape of it.** For a prompt of `n` tokens and model dimension `d`, self-attention
computes

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

where `Q`, `K`, `V` are `n × d` matrices. The `Q·Kᵀ` term is `n × n` — prefill cost grows
with the **square** of the prompt length.

**Do we use it?** Yes, and it is the *only* phase we have. Our classifier runs exactly one
prefill — the customer's sentence through 12 transformer layers — and then reads a label
off the final layer. There is no second phase.

```
measured: p50 9.0 ms, p95 10.1 ms for one sentence (117.7M params, fp32, 4 CPU threads)
```

`√d_k` is a detail worth understanding: without it, the dot products grow with dimension,
softmax saturates, and gradients vanish. Dividing by `√d_k` keeps the variance of the
scores near 1 whatever the head size.

### 2. Decode

**What it is.** The second phase of generation: produce one token, append it, feed it back,
produce the next. Each step is a forward pass over a single new token. It is
**memory-bandwidth-bound**, not compute-bound — you move all the model's weights through
the memory bus to compute one token, so the GPU sits mostly idle.

**Do we use it?** **No.** We emit a label, not a sentence. There is no loop, no
autoregression, nothing to decode. This matters because items 2, 5, 8 and 10 on the list
exist only to make decode faster.

### 3. KV cache

**What it is.** During decode, every new token attends to all previous tokens. Recomputing
the keys and values for the whole history at every step would be `O(n²)` work repeated `n`
times. So `K` and `V` for each past token are computed once and **cached**; each step
computes `Q` for the new token only and attends against the cached `K`/`V`. This turns
decode from `O(n²)` per step into `O(n)`.

**Memory cost**, which is why PagedAttention exists:

```
bytes = 2 (K and V) × layers × heads × head_dim × seq_len × batch × dtype_bytes
```

For a 7B model at 4k context in fp16 that is gigabytes per sequence.

**Do we use it?** **No — and this is the term most often misapplied.** A KV cache is a cache
*inside one generation*, across decode steps. We have no decode steps.

**What we DO have, which is a different thing wearing a similar name:** a result cache in
`cx/engine/client.py`. It maps a normalised sentence to the label the model produced:

```python
_CACHE: OrderedDict[str, dict]   # "my bills" -> {"intent": "bill_due", ...}
CACHE_MAX = 4096
```

```
measured: 6 repeated questions, 5.12 s cold -> 0.49 s warm (10.4x)
```

That is **memoisation**, not a KV cache. It is worth being precise, because the two have
different invalidation rules: a KV cache is valid only within one sequence and is discarded
after; our result cache is valid as long as the model is unchanged, and it caches the
**label and never the answer** — the answer depends on the customer's ledger and the time
of day, and caching that would serve one person's balance to the next.

### 4. vLLM

**What it is.** An inference *server* for generative LLMs, whose central contributions are
PagedAttention and continuous batching. It exists to serve many concurrent generations
efficiently on a GPU.

**Do we use it?** **No**, and it would not help. vLLM serves decoder-only generative models;
ours is an encoder with no generation, running on CPU. Installing it would add a dependency
that cannot be invoked.

**Where it WOULD apply in this codebase:** if the assistant ever calls a local generative
model — for free-text answers, or the Laya-style reasoning sidecar — that model would be a
genuine vLLM candidate, and everything in items 3, 5, 8 and 10 would start to matter.

### 5. PagedAttention

**What it is.** vLLM's core idea, borrowed from operating-system virtual memory. The KV
cache is normally one contiguous block per sequence, sized for the *maximum* length the
sequence might reach — so a request that generates 50 tokens of a possible 2,000 wastes 97%
of its allocation. PagedAttention splits the cache into fixed-size **blocks** (pages) held
in a pool, with a per-sequence **block table** mapping logical positions to physical blocks.
Sequences allocate pages as they grow, and identical prefixes can **share** pages
copy-on-write.

Reported effect: KV memory waste from ~60–80% down to under 4%, so far more sequences fit
at once.

**Do we use it?** **No.** It is a KV-cache memory manager and we have no KV cache.

### 6. Continuous batching

**What it is.** Naive ("static") batching groups N requests, runs them together, and returns
when the *slowest* finishes — so a request generating 10 tokens waits for one generating
500. **Continuous** (or in-flight) batching works at the granularity of a decode *step*: when
a sequence finishes, it leaves the batch immediately and a queued request takes its slot on
the next step.

**Do we use it?** **Not as described** — again, it operates on decode steps. But its honest
cousin, **dynamic micro-batching**, does apply, and I measured it:

```
batch  1    38.0 ms/batch    26.3 sentences/sec
batch  4    84.0 ms/batch    47.6 sentences/sec
batch  8   116.7 ms/batch    68.5 sentences/sec
batch 32   432.1 ms/batch    74.1 sentences/sec
```

Throughput rises **2.8x** from batch 1 to batch 32; per-request latency rises with it. That
is the whole trade in one table. **We have not adopted it, deliberately:** this surface
serves one customer at a time, so there is no queue to batch, and adding batching would add
waiting-for-a-batch latency to every request to buy throughput nobody is asking for.

Revisit the moment there are concurrent users.

### 7. Chunked prefill

**What it is.** A long prompt's prefill is one huge compute-bound pass that blocks every
decode step waiting behind it — so other users' tokens stall. Chunked prefill splits the
prompt into fixed-size chunks and interleaves them with decode steps, trading a little
prefill throughput for much better tail latency for everyone else.

**Do we use it?** **No.** Our prompts are single sentences, capped at 48 tokens. There is
nothing long enough to chunk, and no decode steps to interleave with.

### 8. Prefix caching

**What it is.** Many requests share a long leading prefix — a system prompt, a few-shot
preamble, a document. Prefix caching stores the KV cache **for that prefix** and reuses it
across requests, so the shared part is prefilled once rather than per request. With
PagedAttention this is page sharing; Anthropic's API exposes the same idea as prompt
caching.

The saving is exactly the prefill of the shared prefix:

```
saved ≈ (prefix_tokens / total_tokens) of prefill cost, per cache hit
```

**Do we use it?** **Not in the classifier** — our inputs are short and share no prefix.
**Yes elsewhere in this codebase:** the ACE work uses Anthropic prompt caching for exactly
this reason, and if a generative assistant is added here with a long system prompt, this is
the first optimisation to reach for.

### 9. Speculative decoding

**What it is.** Decode is memory-bandwidth-bound, so a big model is idle most of each step.
A small **draft** model proposes `k` tokens cheaply; the big **target** model then verifies
all `k` in ONE forward pass and accepts the longest correct prefix. Rejected tokens are
discarded and the target's own token is used. The output distribution is provably identical
to the target model's — this is a pure latency optimisation, not an approximation.

Expected speed-up with draft acceptance rate `α` and draft length `k`:

```
speedup ≈ (1 - α^(k+1)) / ((1 - α)(k·c + 1))        c = draft cost / target cost
```

**Do we use it?** **No.** It accelerates decode; we have none. There is a tempting analogy —
our cheap char-n-gram `neighbours.py` "drafting" an intent for the transformer to verify —
but it is only an analogy: we use neighbours as a *fallback when the model is unsure*, not
as a draft the model verifies, and verification would cost the same forward pass we are
trying to avoid.

### 10. Quantization

**What it is.** Store and compute weights in fewer bits. fp32 → int8 is 4x smaller and, on
hardware with integer kernels, can be faster. The standard affine scheme:

```
scale  = (max - min) / (2^b - 1)
zero   = round(-min / scale)
q      = clamp(round(x / scale) + zero, 0, 2^b - 1)
x̂      = (q - zero) × scale          # dequantised, lossy
```

**Dynamic** quantization quantizes weights ahead of time and activations on the fly;
**static** calibrates activation ranges with sample data; **QAT** simulates the rounding
during training so the model learns to tolerate it.

**Do we use it? NO — and this is the most useful thing in this document, because I expected
it to win and measured the opposite.**

```
fp32   p50  9.0 ms   p95 10.1 ms   accuracy 87.1%
int8   p50 17.2 ms   p95 25.0 ms   accuracy 84.9%     <- 1.9x SLOWER, 2.2pp worse
agreement between the two: 454/489 (92.8%)
```

**Why it lost.** Dynamic quantization pays a quantize/dequantize cost per linear layer, and
wins only when the matrix multiply is large enough for int8 kernels to repay it. Our
matrices are small — batch 1, ≤48 tokens — so the overhead dominates, while fp32 is already
running on well-tuned MKL kernels. Quantization pays off on large batches, long sequences,
and memory-bandwidth-bound workloads. Ours is none of those.

The 92.8% agreement is the other half of the lesson: quantization is **lossy**, and on 7% of
sentences it changed the answer.

### 11. Latency

**What it is.** For generative serving the field distinguishes **TTFT** (time to first
token, dominated by prefill) from **TPOT/ITL** (time per output token, dominated by decode),
with

```
total ≈ TTFT + TPOT × output_tokens
```

**For us there is only one number**, because there is one forward pass — and the thing that
matters is the distribution, not the mean. p95 is what the unlucky customer experiences.

Measured end to end, per turn:

```
classifier forward pass      p50   9 ms
Laya second opinion          p50 ~860 ms, tail several seconds   (advisory, never blocking)
OCR on a screenshot              ~9 s first call (model load), ~2-3 s after
full /api/cx/ask turn        p50 ~0.8 s, dominated by ledger reads, not the model
```

**The model is not the bottleneck.** That is why the engineering effort went into the result
cache and into not calling Mongo twice, rather than into making a 9 ms forward pass faster.

---

## Part 2 — the formulas actually running in this system

### Softmax — turning scores into probabilities

The final layer emits one raw score (**logit**) per intent. Softmax maps them to a
distribution:

```
p_i = exp(z_i) / Σ_j exp(z_j)
```

In practice `max(z)` is subtracted from every logit first — mathematically identical,
numerically necessary, since `exp(1000)` overflows.

This is the `confidence` in every answer, and `CONFIDENT` in the engine is a threshold on
it. **It is a confidence, not a probability of being right** — a point the codebase is
deliberate about, because a model can be 98% confident and wrong, as ours was on
`"i paid alex 100"`.

### Categorical cross-entropy — the training loss

For one example with true class `y`:

```
L = -log(p_y)
```

Correct and certain (`p_y = 1`) costs 0. Correct and unsure (`p_y = 0.5`) costs 0.69.
Confidently wrong (`p_y = 0.01`) costs 4.6. The penalty for confident error is unbounded,
which is exactly the right shape.

Random guessing over `C` classes gives `L = ln(C)`. With our 26 intents that is **3.258** —
which is why "loss 3.2581 at step 100" in the training log meant *nothing has been learned
yet*, and why watching it fall to 0.06 meant it had.

### Class weighting — our loss is not plain cross-entropy

Intents are wildly unbalanced: `out_of_scope` has thousands of rows, `disputed` has 202. A
plain average would let the model ignore rare classes. So each class carries a weight
inversely proportional to its frequency:

```
L = -w_y · log(p_y)        w_c ∝ 1 / count(c)
```

From a real run:

```
disputed 5.03   spend_rank 4.59   spend_analysis 4.11   out_of_scope 1.17
```

`out_of_scope` additionally carries a **2.0x policy weight** on top of its frequency weight
— not because it is rare, but because answering something we should have declined is worse
than the reverse. That is a product judgement expressed as a number in the loss.

### Why not BCE

**Binary cross-entropy** is the two-class case, applied per output:

```
L = -[ y·log(p) + (1-y)·log(1-p) ]
```

With a sigmoid per class rather than one softmax, BCE gives **multi-label** classification:
a sentence can be two intents at once, and the probabilities need not sum to 1. Our trainer
supports it behind `--multilabel` and we do not use it: a turn is routed to exactly one
handler, so the mutual exclusivity softmax enforces is a property we want, not a limitation.

### Gradient descent and AdamW

Training adjusts every weight against the slope of the loss:

```
θ ← θ - η · ∂L/∂θ
```

`∂L/∂θ` comes from **backpropagation** — the chain rule applied backwards through the
network. `η` is the learning rate.

Plain SGD uses one `η` for every weight. **Adam** keeps two running averages per weight:

```
m_t = β₁·m_{t-1} + (1-β₁)·g          # momentum: where the gradient has been heading
v_t = β₂·v_{t-1} + (1-β₂)·g²         # variance: how noisy this weight's gradient is
m̂ = m_t/(1-β₁ᵗ),  v̂ = v_t/(1-β₂ᵗ)   # bias correction, since both start at 0
θ ← θ - η · m̂ / (√v̂ + ε)
```

A weight with consistent gradients takes big steps; a noisy one takes small ones.
**AdamW** — what we use — differs from Adam in one detail that matters: weight decay is
applied directly to the weights rather than added to the gradient,

```
θ ← θ - η · m̂/(√v̂ + ε) - η · λ · θ          (λ = 0.01 here)
```

because folding decay into the gradient lets Adam's per-weight scaling distort the
regularisation, which is the "W" in AdamW.

### OneCycleLR — the warm-up you saw in the logs

The learning rate is not constant. It rises from near zero to `max_lr` over the first 10%
of steps (`pct_start=0.1`), then anneals down.

Warm-up matters because a randomly initialised classifier head on top of a pretrained
encoder produces large, meaningless gradients at first; taking full-size steps on them
damages good pretrained weights. This is why **loss barely moved for the first 200 steps of
every run** and that was correct, not a stall.

```
step 100   loss 3.2597      <- warm-up, LR still climbing (random = 3.258)
step 300   loss 3.0749      <- learning begins
step 1700  loss 0.1525
```

### Gradient clipping

```
if ‖g‖ > c:  g ← g · c/‖g‖          (c = 1.0)
```

One freak batch can produce an enormous gradient and destroy the weights. Clipping caps the
step size while keeping its direction.

### Cosine similarity and TF-IDF — `neighbours.py`

For the nearest-neighbour fallback we represent a sentence as a sparse vector of
character 3-, 4- and 5-grams.

**Term frequency, sublinear:** `tf = 1 + log(count)` — a gram appearing 10 times is not 10
times as important.

**Inverse document frequency:**

```
idf(g) = log( (N + 1) / (df(g) + 1) ) + 1
```

A gram in every document carries no information and gets a weight near 1; a rare one scores
high. (This is precisely what the ACE retrieval bug was: a `Lexical()` scorer with **no
IDF**, so "wallet" — present in every document of a wallet repo — contributed as much as a
distinctive term, and ranking went flat.)

**Cosine similarity** — the angle between two vectors, ignoring their length:

```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

Length-independence is the point: a long sentence and a short one about the same thing
should still match. Vectors are L2-normalised at index time, so the runtime cost is just the
dot product.

**Why character n-grams rather than words:** `details` and `detials` share almost every
3-gram, so a typo stays near its correct spelling for free — and no tokeniser is needed,
which matters when Devanagari and Arabic word-splitting have already caused two production
bugs here.

**The decision rule**, which is as important as the similarity:

```
FLOOR  = 0.34     best neighbour must be at least this close
MARGIN = 0.18     winner's share of the weighted vote must lead the runner-up by this
```

Below either, it returns nothing. A mixed neighbourhood is not an answer.

### Levenshtein distance — typo tolerance in the engine

The minimum number of single-character insertions, deletions or substitutions to turn one
string into another, by dynamic programming:

```
d[i][j] = min( d[i-1][j] + 1,                  # delete
               d[i][j-1] + 1,                  # insert
               d[i-1][j-1] + (a[i] ≠ b[j]) )   # substitute
```

Ours is **bounded** — it abandons a comparison as soon as the best possible result exceeds
the limit — and the limit is length-dependent: 1 edit for words of 6 characters or fewer, 2
above. Two edits on a four-letter word reaches most of the dictionary.

### McNemar's test — deciding whether a new model is actually better

Two models on the same examples are **paired** data, so comparing two accuracy percentages
is the wrong test. What matters is only the examples where they **disagree**: `a` that the
old model got right and the new one wrong, `b` the reverse. Under the null hypothesis that
they are equally good, each disagreement is a coin flip:

```
p = 2 · Σ_{i=0..min(a,b)} C(n, i) / 2ⁿ          n = a + b
```

This is why v8 was rejected despite scoring higher overall, and why v10's apparent drop was
reported as a draw:

```
v7 vs v8    19 / 12    p = 0.281    not significant
v7 vs v9    18 / 19    p = 1.000    dead heat
v9 vs v10   21 / 14    p = 0.311    not significant
```

In all three cases the accuracy difference was noise. The decisions were made on **leaks**
and on **behaviour through the engine** instead — which is the real lesson: pick the metric
that matches the harm, not the one that is easiest to compute.

---

## Part 3 — what we changed because of this, and what we did not

| technique | applies | status |
|---|---|---|
| Prefill | yes — our only phase | 9 ms p50, measured |
| Decode | no — we emit a label, not text | — |
| KV cache | no — no decode steps | we have a *result* cache, 10.4x on repeats |
| vLLM | no — encoder, not generative | revisit if an LLM is added |
| PagedAttention | no — no KV cache to page | — |
| Continuous batching | the micro-batching cousin does | measured 2.8x throughput, **not adopted**: one user at a time |
| Chunked prefill | no — 48-token inputs | — |
| Prefix caching | not here; yes in ACE | first thing to add with a system prompt |
| Speculative decoding | no — no decode to speculate on | — |
| Quantization | measurable, and it **lost** | fp32 kept: int8 was 1.9x slower, 2.2pp worse |
| Latency | yes | model is 9 ms; the ledger dominates the turn |

**The honest summary: of the eleven, four describe phases or caches we genuinely have or
have an analogue of (prefill, result caching, micro-batching, latency), one was measured and
rejected on the evidence (quantization), and six belong to generative decoding, which this
model does not do.**

Adopting a technique that does not fit is not neutral — it adds a dependency, a failure
mode, and a claim that will not survive the first hard question. Measuring it and writing
down why it did not fit is the part worth keeping.

---

## Part 4 — the generative layer, and why it is switched off

Added after the encoder sections above, and it is the reason most of Part 1 stopped being
hypothetical: `cx/phrase.py` runs **Qwen2.5-Instruct**, 4-bit quantized (GGUF Q4_K_M), on
CPU through llama.cpp. It is the first **autoregressive** component in this system, so it
has a real prefill phase, a real decode loop and a real KV cache.

### The rule

**The model phrases. It never decides and never computes.** The engine resolves the intent,
reads the ledger, applies the guards and produces FACTS. The model turns those into a
sentence. Then every claim in that sentence is checked back against the facts, and a
failure means the template is used instead.

This is not a quality gate. It is the condition under which a generative model is allowed
near a payments surface at all.

### What the phases actually cost here

```
prefill          178-token system prompt + facts
decode           0.5B  24.8 ms/token      1.5B  47.2 ms/token
TTFT cold        0.5B   743 ms (p50)      1.5B  1256 ms (p50)
prefix caching   1918 ms -> 275 ms  (7.0x) — the system prompt is byte-identical every
                 call, so llama.cpp keeps its KV and re-prefills only the facts
quantization     Q4_K_M: 491 MB / 1117 MB instead of ~2 GB / ~6 GB. A genuine win here,
                 unlike int8 on the encoder (Part 1, item 10), because decode is
                 memory-bandwidth-bound and 4-bit weights move four times faster
```

### Three layers of verification, each added after a real failure

**1. Grounding — a number the engine never produced.**

```
numbers(output) ⊆ numbers(facts)
```

No tolerance, no rounding. "About AED 4,600" is rejected when the figure is AED 4,634.

**2. Contradiction — the same digits, the opposite meaning.** Grounding cannot see this,
because "to Alex" and "from Alex" contain identical numbers:

```
draft:   "Your 100 AED TO Alex ... is delivered."
0.5B:    "That's AED 100.00 FROM Alex ..."
```

Pairs checked: to/from, sent/received, delivered/failed, more/less, week/month. On a
payments surface the direction of the money IS the fact.

**3. Role and speaker — who did it.** This one only appeared with the BIGGER model, which
writes well enough to make a confident claim about the wrong person:

```
draft:   "Alex Morgan — AED 3,276.00 across 57 payments"
1.5B:    "Alex Morgan SPENT AED 3,276.00 on 57 transactions"   <- the customer spent it

draft:   "I don't have a figure from this conversation to check."
1.5B:    "YOU don't have a figure from this conversation to check."
```

### The measurement, and the decision

```
                  passed checks   TTFT p50   TPOT p50
Qwen2.5-0.5B           7/8          743 ms   24.8 ms/tok
Qwen2.5-1.5B           6/8         1256 ms   47.2 ms/tok
```

**The bigger model scores worse, and reading the outputs is the only way to understand
why.** It does not write worse — it writes better, and therefore attempts claims the small
model never reaches for. The 0.5B's single failure was a crudely invented number, caught by
the first and simplest check. The 1.5B's two failures were a swapped subject and a swapped
speaker, each of which needed a check that did not exist until it produced one.

So the raw score inverts the real quality ordering, and anyone reading only the number
would reach the wrong conclusion.

**It is off by default** (`CX_PHRASE=1` to enable) and should stay off. Of eight real
answers most rewrites were identical to the template or slightly worse, one was
ungrammatical, and the model that writes best is the model that lies best. A layer whose
main achievement is being correctly rejected a quarter of the time has not earned 750 ms.

What it earned instead is this section, a verification harness that will outlive the
experiment, and one lesson worth more than the feature: **fluency is not accuracy, and the
checks you are able to write decide what you are allowed to ship.**

### A note on testing it

`cx/tests/test_phrase.py` runs the three checks **without loading the model**. A
verification layer that only works when a 491 MB file is present is a verification layer
nobody runs. Writing the tests immediately found a real hole: the phrase matcher stripped
digits, so "30 days" could never match and `"the last 30 days"` rewritten as `"this week"`
passed the contradiction check untouched.
