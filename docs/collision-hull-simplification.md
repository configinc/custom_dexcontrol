> Historical measurements below used the previous synchronous full-path guard.
> The runtime now defaults to 1 mm outer hulls with a separate monitor and
> endpoint-approval worker. The comparison script now measures endpoint policy;
> old full-path timing columns are not current writer timings. See
> [control-architecture.md](control-architecture.md).

# Bounded outer-hull comparison

The offline comparison favors **0.5 mm per shape as the first candidate**: it
removes 81.5% of vertices while retaining most of the measured safety-loop speed
benefit. This experiment generates candidate collision assets; it does not change
the server configuration or the running Pose Lab model.

## Method and geometric contract

`examples/advanced_examples/collision_hull_tools.py` constructs an outer convex
polytope for each existing connected-component hull. Support planes are placed
outside all original vertices. Intersections that exceed the requested outward
error cause additional support directions to be added. A 1 um coordinate grid
removes numerical slivers before triangulation; support-based outward scaling
restores containment after that rounding. The final error includes both steps.

Every result must pass two checks, both before export and after reloading the
OBJ/URDF through the actual Pinocchio/Coal model loader:

1. Every original vertex satisfies every outer face halfspace, with a 1 nm
   numerical tolerance. This checks containment of the whole original convex solid.
2. Every outer vertex is within the specified Euclidean distance of the original
   solid. Distance to a convex set is a convex function, so this bounds the error
   over the whole outer solid, including edges and faces.

Both meshes must be closed, consistently oriented convex solids. Geometry
placements, joint names, connected components and collision pairs must match
after reloading. A failed check stops the experiment; no alternate geometry is
silently substituted. This is an adaptive approximation, not an optimizer that
guarantees the smallest possible vertex count. Lower-detail input meshes may
even gain vertices, although every Vega component shrank in this run.

With outward error `e` on each of two separated shapes, their computed clearance
can shrink by up to `2e`. Shape simplification does not reduce the configured
10 mm margin. These historical measurements exclude external grippers, cables and payloads;
the current runtime additionally includes Robotiq envelopes.
The existing wheel and SRDF exclusions are unchanged.

Algorithm references: [SciPy halfspace intersections](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html)
and [trimesh nearest-surface queries](https://trimesh.org/trimesh.proximity.html).
`rtree` is needed only for the offline construction/verification queries.

## Measurements on 2026-10-30

Intel Core i9-14900F, Python 3.12.13, Coal 3.0.2, Pinocchio 3.9.0, NumPy 2.3.5,
trimesh 5.2.0, dexmate-urdf 0.8.4. BLAS/OpenMP thread counts were set to one.
All four models have 38 shapes, 392 shape pairs, 158 link pairs and 20 joints.

The experiment samples 5,000 configurations clear under the original model
(seed 42). Every model receives the identical measured pose, measured velocity,
target and feedforward velocity. Measurement order rotates between variants.
Clipping, the 10 mm clearance, 50 ms forecast and 32-query budget are identical.

| Per-shape outward error | Total vertices | Maximum verified outward error | Full-distance median | Full-distance p99 |
| --- | ---: | ---: | ---: | ---: |
| Original | 96,528 | 0 | 0.973 ms | 1.222 ms |
| 0.5 mm | 17,894 | 0.477 mm | 0.599 ms | 0.740 ms |
| 1 mm | 10,655 | 0.951 mm | 0.516 ms | 0.634 ms |
| 2 mm | 6,652 | 1.897 mm | 0.453 ms | 0.552 ms |

"Full distance" includes FK and precise Coal queries for every enabled pair,
as used by the viewer/export. The real safety path also performs clipping,
checks current clearance, and verifies target and velocity-forecast segments:

| Per-shape outward error | Safety median | Safety p95 | Safety p99 | Maximum | Checks over 5 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0.495 ms | 2.915 ms | 3.989 ms | 6.293 ms | 8 |
| 0.5 mm | 0.469 ms | 2.681 ms | 3.439 ms | 4.808 ms | 0 |
| 1 mm | 0.460 ms | 2.626 ms | 3.312 ms | 4.559 ms | 0 |
| 2 mm | 0.453 ms | 2.568 ms | 3.192 ms | 4.286 ms | 0 |

The exact-distance improvement is larger than the total safety improvement.
The safety path already skips most distant pairs with AABBs and retains FK,
Python orchestration and adaptive path subdivision overhead. At 0.5 mm, median
full-distance time falls by 38.4%, while median safety time falls by 5.2% and
safety p99 by 13.8%.

Larger solids can also change subdivision and rejection behavior. To avoid
mistaking an early rejection for a speed gain, the report separately compares
the same 4,596 commands accepted by all four models. Their safety median/p99
are 0.479/2.851 ms originally, 0.459/2.601 ms at 0.5 mm, 0.452/2.604 ms at 1 mm,
and 0.448/2.603 ms at 2 mm. The extra simplification beyond 0.5 mm brings little
additional benefit for this common accepted set.

Of 4,611 originally accepted commands, the 0.5 mm / 1 mm / 2 mm models reject
5 / 8 / 15 additional commands. The first two variants' additional rejections
are all inability to certify paths within the unchanged query budget. At 2 mm,
14 are budget rejections and one is below the clearance margin. None of the
originally rejected commands became accepted in this sample. Budget rejections
must not be interpreted as measured collisions.

Across 790,000 link-pair queries per variant, maximum clearance reductions were
0.907 / 1.760 / 3.559 mm, within twice each requested outward-error bound. The
largest numerical clearance increase was 0.198 um, below the existing Coal
distance tolerance of 1 um. All 114 exported and reloaded shapes passed the
containment and outward-error checks.

These are offline computation times, not a demonstrated 200 Hz deadline.
SDK decoding, IK, scheduling, publication and physical braking are excluded.
Zero samples exceeding 5 ms does not establish a worst-case bound on the robot PC.

## Preset clearances

Presets are bounded to the same URDF joint limits as Pose Lab. Values are model
surface separations, not physical measurements or complete action-path checks.

| Preset | Original | 0.5 mm | 1 mm | 2 mm |
| --- | ---: | ---: | ---: | ---: |
| Folded | 6.929 mm | 6.707 mm | 6.708 mm | 5.367 mm |
| Folded arms / upright torso | 20.837 mm | 20.553 mm | 20.460 mm | 19.452 mm |
| Initial arms / upright torso | 72.011 mm | 71.929 mm | 71.489 mm | 70.598 mm |
| Lifted arms / upright torso | 49.357 mm | 49.354 mm | 49.353 mm | 49.353 mm |

Full Folded remains below the configured 10 mm margin in every variant.
Its nearest pair remains `base / L_arm_l6`. Different error settings generate
independent approximations; they are not necessarily nested inside one another.

## Reproduce

Install the optional offline dependency into the existing dexcontrol environment:

```bash
pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python examples/advanced_examples/benchmark_collision_hulls.py \
  --errors-mm 0.5 1 2 --iterations 5000 --seed 42 \
  --output-dir /tmp/vega-hull-comparison
```

Use a new output directory. It contains `report.json`, the identical input arrays
in `inputs.npz`, and separate collision URDFs/OBJ files for each error setting.
Reports include per-shape verification, preset distances, timing distributions,
accept/reject transitions, and the subset of commands accepted by every model.
The generated URDF references local vendor assets and requires the original SRDF;
the directory is not a self-contained deployment package.

The measured run is saved locally at `/tmp/vega-outer-hulls-5000/report.json`.
The live server and Pose Lab continue to use the original vendor-component hulls.
