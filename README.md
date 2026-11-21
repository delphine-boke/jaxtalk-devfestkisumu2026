# JAX, but start where NumPy ends

DevFest Kisumu 2026 · talk notes

## What we have

A beginner-to-intermediate talk on JAX as a transform engine, not another PyTorch.

- **Talk accepted.** DevFest Kisumu, 21 November 2026. Slot is 30–60 minutes. Plan is about 30 minutes of talk plus live cells, then questions.
- **Tags.** AI – JAX, AI.
- **Notebook.** Step-by-step cells for Kaggle. Works on TPU or GPU. One function (`selu`) through immutability, PRNG keys, `grad`, `vmap`, and `jit`.
- **Slides.** 12 slides. Google green on light gray, matched to the DevFest template. Speaker notes on each slide.
- **No client work.** Public teaching material only.

The notebook is the demo. The slides are the spine.

## What the room leaves with

- `jnp` looks like NumPy, until the array will not mutate.
- Randomness is a key you thread, not a global generator.
- One function, composed: `jit(grad(vmap(f)))`.
- First call is the compile tax. Second call is the point.
- A decide-rule for JAX versus PyTorch.

## Real examples

These are places JAX is actually used. They are not a claim that JAX replaced PyTorch. It did not. Most product serving is still on other stacks. JAX is the research and scientific-computing stack where the transforms pay.

### AlphaFold

DeepMind. Predicts protein structure from an amino-acid sequence. The 2024 Nobel Prize in Chemistry was for this work. The public database has been used by over 3 million researchers in more than 190 countries, and holds predictions for 200 million-plus structures. AlphaFold 3’s inference code is JAX.

### AlphaGenome 2

DeepMind, November 2026. A DNA sequence model that predicts what a genetic variant does in a specific cell type, not in a whole tissue. Same family of stack: JAX, transforms, TPU.

### Escalante

A protein-design company. A design is scored on several models at once (bind, soluble, stable, expressible), including an AlphaFold 2 implementation in JAX. They spin up 2,000 to 4,000 spot TPUs, run the job for about half an hour, and shut them down. On those jobs, TPU v6e was reported at 3.65× better performance per dollar than an H100.

## When JAX, when not

**Reach for JAX** when you want `grad`, `vmap`, and `scan` to compose, or one codepath from CPU to GPU to TPU.

**Stay on PyTorch** when the ecosystem is the constraint (Hugging Face trainers, many SOTA repos), when you need eager debugging and dynamic graphs first, or when the team has no XLA muscle yet.

## Learn in this order

1. JAX 101: `jnp`, keys, `jit`, `grad`, `vmap`.
2. One loop: `jit(value_and_grad)` plus Optax on a tiny MLP.
3. Flax NNX.
4. 201: `scan`, sharding, Orbax.

## Files

- `DevFest Kisumu 2026 - JAX, but start where NumPy ends` — the slide deck.
- `jax-notebook-demo-devfestksm2026.ipynb` — the Kaggle notebook.

## Links

- Slides: https://docs.google.com/presentation/d/1at1DkbLNAXoNOUq0otpuFsw0_bFZczHODmpn8BIpRHs/edit?usp=sharing

- https://docs.jax.dev

- https://docs.jaxstack.ai


