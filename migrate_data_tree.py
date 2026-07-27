#!/usr/bin/env python3
"""
Migrate an existing data tree from the v1 naming layout to v2.

v1 -> v2 mapping
----------------
  tau_maps/fullFT_tau_reconstruction/tau_map_<fb>.npy
      -> tau_maps/truth/gas/tau_<fb>.npy
  tau_maps/fullFT_tau_reconstruction/tau_map_dm_<fb>.npy
      -> tau_maps/truth/dm/tau_<fb>.npy
  tau_maps/2D_FT_upgrade_tau_reconstruction/tau_map_<fb>_reconstructed_<sel>.npy
      -> tau_maps/recon/gas/tau_<fb>_<sel>.npy
  tau_maps/2D_FT_upgrade_tau_reconstruction/tau_map_dm_<fb>_reconstructed_<sel>.npy
      -> tau_maps/recon/dm/tau_<fb>_<sel>.npy

  <proj>_CAP_code/<fb>[_reconstructed]/<method>/<tracer>/<sel>/tau_apertures.npz
      -> aperture_photometry/<proj>/<fb>/<truth|recon>/<tracer>/<sel>/tau_apertures.npz

  tau_map_prefactors/  -> tau_prefactors/

  delta_fields/, halo_indices/  -> unchanged (these already stripped the
  '_reconstructed' suffix in v1, so their filenames are identical in v2).

Anything matching 2D_FT_massdep is reported and left alone - that method was
removed, so those products are orphans you can delete once you have checked.

Usage
-----
    python migrate_data_tree.py /path/to/data                # dry run
    python migrate_data_tree.py /path/to/data --apply        # do it
    python migrate_data_tree.py /path/to/data --apply --copy # copy, don't move
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

FEEDBACKS = ("fiducial", "strongest_AGN")

SOURCE_OF = {
    "fullFT_tau_reconstruction": "truth",
    "2D_FT_upgrade_tau_reconstruction": "recon",
}

MASSDEP = "2D_FT_massdep_tau_reconstruction"


def plan_tau_maps(sim_root: Path):
    """Yield (old, new) for tau maps under <sim_root>/tau_maps/."""
    tau_root = sim_root / "tau_maps"
    if not tau_root.is_dir():
        return
    for method_dir in sorted(tau_root.iterdir()):
        if not method_dir.is_dir():
            continue
        source = SOURCE_OF.get(method_dir.name)
        if source is None:
            continue  # massdep / already-migrated dirs handled elsewhere
        for old in sorted(method_dir.glob("*.npy")):
            stem = old.stem
            if stem.startswith("tau_map_dm_"):
                tracer, rest = "dm", stem[len("tau_map_dm_"):]
            elif stem.startswith("tau_map_"):
                tracer, rest = "gas", stem[len("tau_map_"):]
            else:
                print(f"  !! unrecognised tau-map name, skipped: {old}", file=sys.stderr)
                continue

            # rest is '<feedback>[_reconstructed][_<seltag>]'
            feedback = next((f for f in FEEDBACKS if rest.startswith(f)), None)
            if feedback is None:
                print(f"  !! unknown feedback in: {old}", file=sys.stderr)
                continue
            tail = rest[len(feedback):]
            tail = re.sub(r"^_reconstructed", "", tail)  # drop redundant marker

            new = tau_root / source / tracer / f"tau_{feedback}{tail}.npy"
            yield old, new


def plan_ap_outputs(sim_root: Path):
    """Yield (old, new) for aperture-photometry outputs."""
    for cap_dir in sorted(sim_root.glob("*_CAP_code")):
        projection = cap_dir.name[: -len("_CAP_code")]
        for old in sorted(cap_dir.rglob("tau_apertures.npz")):
            rel = old.relative_to(cap_dir).parts
            if len(rel) != 5:
                print(f"  !! unexpected AP depth, skipped: {old}", file=sys.stderr)
                continue
            gas_type, method, tracer, seltag, fname = rel
            source = SOURCE_OF.get(method)
            if source is None:
                continue
            feedback = re.sub(r"_reconstructed$", "", gas_type)
            new = (
                sim_root / "aperture_photometry" / projection / feedback
                / source / tracer / seltag / fname
            )
            yield old, new


def plan_prefactors(sim_root: Path):
    old_dir = sim_root / "tau_map_prefactors"
    if not old_dir.is_dir():
        return
    for old in sorted(old_dir.glob("*.npy")):
        yield old, sim_root / "tau_prefactors" / old.name


def find_orphans(sim_root: Path):
    for p in sim_root.rglob("*"):
        if p.is_file() and MASSDEP in str(p):
            yield p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", type=Path, help="the directory holding <sim>/ subdirs")
    ap.add_argument("--apply", action="store_true", help="perform the migration")
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    args = ap.parse_args()

    if not args.data_root.is_dir():
        sys.exit(f"No such directory: {args.data_root}")

    moves, collisions = [], []
    for sim_root in sorted(p for p in args.data_root.iterdir() if p.is_dir()):
        print(f"\n=== {sim_root.name} ===")
        for planner in (plan_tau_maps, plan_ap_outputs, plan_prefactors):
            for old, new in planner(sim_root):
                if new.exists() or new in {n for _, n in moves}:
                    collisions.append((old, new))
                else:
                    moves.append((old, new))
                print(f"  {old.relative_to(args.data_root)}"
                      f"\n    -> {new.relative_to(args.data_root)}")
        orphans = list(find_orphans(sim_root))
        for o in orphans:
            print(f"  [ORPHAN massdep, left in place] {o.relative_to(args.data_root)}")

    print(f"\n{len(moves)} file(s) to migrate, {len(collisions)} collision(s).")
    if collisions:
        print("COLLISIONS - resolve these before applying:")
        for old, new in collisions:
            print(f"  {old} -> {new}")
        sys.exit(1)

    if not args.apply:
        print("\nDry run. Re-run with --apply to perform the migration.")
        return

    op = shutil.copy2 if args.copy else shutil.move
    for old, new in moves:
        new.parent.mkdir(parents=True, exist_ok=True)
        op(str(old), str(new))
    print(f"Done: {len(moves)} file(s) {'copied' if args.copy else 'moved'}.")

    if not args.copy:
        for sim_root in sorted(p for p in args.data_root.iterdir() if p.is_dir()):
            for d in sorted(sim_root.rglob("*"), reverse=True):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()


if __name__ == "__main__":
    main()