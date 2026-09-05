# Evolution Showcase: Design Spec

Date: 2026-09-05
Status: Approved, building

## 1. Goal

Make the project's headline finding **watchable**: predation drives brain
complexity. Two worlds run side by side from the same seed, differing only in
whether agents can attack each other, and the viewer sees their brains diverge.

Success criterion: someone with no context can watch for thirty seconds and
correctly state what the experiment showed.

## 2. Hard constraint: everything shown must be real

The browser **replays**; it never simulates. Every frame is recorded output from
the Python engine that passes the 89-test suite, with energy conservation and
seeded determinism. No JavaScript reimplementation of any simulation rule.

This is not a stylistic preference. The project's only real asset is that its
claims are verified. A JS port would be an unverified second engine whose numbers
could silently diverge from `docs/results/`, and the showcase would stop being
evidence of anything.

## 3. Data pipeline

`evolution/showcase.py` runs both worlds and emits one JSON bundle.

```
world A: Config(width=64, height=64, seed=1, allow_attack=True)
world B: Config(width=64, height=64, seed=1, allow_attack=False)
40,000 ticks each, frame captured every 100 ticks -> 400 frames
```

### 3.1 Why the frames are packed

Naive encoding does not fit. Eight fields per agent, ~150 agents, 800 frames,
two worlds is roughly 16 MB before the plant grids, and the plant grids alone at
64x64 per frame would add another 16 MB. The artifact limit is 16 MB total.

| Data | Encoding | Approx cost |
|---|---|---|
| Plant field | downsampled to 32x32, each cell quantised to 0-9, emitted as one 1024-character string | ~1 KB/frame |
| Agents | 4 characters each: `x`, `y` from a 64-character alphabet (grid is 64 wide), diet quantised 0-9, brain size quantised 0-9 | ~600 B/frame |
| Per-frame scalars | population, mean links, mean hidden, mean generation, mean diet | ~40 B/frame |

Roughly 650 KB per world, **~1.3 MB for both**. The bundle must stay under 2 MB;
a test enforces this.

### 3.2 Quantisation is lossy on purpose

Agent positions are exact (the grid is 64 wide and the alphabet is 64
characters). Diet and brain size are quantised to ten buckets, which is ample for
colour and radius and is the reason the bundle fits. The per-frame *scalars* are
carried at full precision, so every number displayed in the HUD is exact; only
the per-dot rendering is quantised.

## 4. Champion brains

At each milestone tick the exporter dumps the complete genome of the
most-connected living agent: nodes with their activations, and connections with
weights and enabled flags. Rendering these shows the same measurement the report
makes in a table -- 10 flat input-to-output links at the start, hidden structure
later -- as an actual network diagram.

Payload is negligible (three genomes per world). Champion genomes must pass
`genome.validate`, which a test asserts.

## 5. Milestones are computed, never authored

Hardcoding "peak complexity at tick 21,000" would rot the moment anything is
retuned. All three are derived from the recorded stats:

- **first_structure**: first tick where `mean_hidden > 0.5`
- **peak_complexity**: tick of `argmax(mean_hidden)`
- **crash**: first tick where population falls more than 40% below its trailing
  2,000-tick maximum; absent if no such tick exists

A milestone that does not occur is omitted rather than faked. The page must
render correctly with zero, one, two, or three milestones present.

## 6. The page

Two canvases side by side sharing one clock. Plant field as a green ground,
agents as dots: hue interpolated along the diet axis (teal at 0, amber at 1),
radius scaled by brain complexity.

Beneath each world a HUD shows population, mean links, and hidden neurons, with
**hidden neurons drawn as a bar** so the divergence reads as a shape rather than
two numbers the viewer must compare mentally.

Controls: play/pause, speed 1x / 4x / 16x, a scrub bar carrying the milestone
markers, and click-to-jump on those markers. `prefers-reduced-motion` starts the
page paused.

A caption strip states the experimental setup in one sentence -- same seed, same
constants, one flag -- because a viewer who does not know that has no reason to
find the divergence interesting.

## 7. Module layout

| File | Responsibility |
|---|---|
| `evolution/showcase.py` | run both worlds, quantise and pack frames, compute milestones, capture champion genomes, write the bundle |
| `tools/build_showcase.py` | inject the bundle into the HTML template, write `docs/results/showcase.html` |
| `tests/test_showcase.py` | exporter invariants |

`showcase.py` imports only from the simulation core and writes JSON. It contains
no HTML and no rendering. The core is not modified.

## 8. Testing

1. **Frame parity.** Both worlds export the same number of frames, so the shared
   clock cannot desynchronise.
2. **Quantisation round-trip.** Decoding a packed frame recovers agent positions
   exactly, and diet within one bucket (0.05).
3. **Size cap.** The bundle is under 2 MB.
4. **Milestone ordering.** Any milestones present are in ascending tick order and
   lie within the run.
5. **Champion validity.** Every captured genome passes `genome.validate`.
6. **Determinism.** Exporting twice with the same seed produces byte-identical
   bundles.

## 9. Known risks

| Risk | Response |
|---|---|
| A world goes extinct before 40,000 ticks | The exporter records short and the page pads the dead world with its final frame, showing "EXTINCT at tick N". Extinction is a real outcome and must be displayed, not hidden. |
| Bundle exceeds 2 MB | Halve the frame rate to every 200 ticks before reducing spatial resolution; temporal resolution is worth less here than the spatial pattern. |
| The two worlds look identical to a casual viewer | The hidden-neuron bar is the mitigation: it is the metric that actually diverges 5-9x, whereas the dot fields genuinely do look similar. |
