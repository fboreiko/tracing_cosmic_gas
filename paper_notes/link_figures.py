#!/usr/bin/env python3
"""Point paper_notes/Figures at the real plots in plots/pme_reconstruction/.

main.tex says \\includegraphics{Figures/<name>.pdf}. Rather than keeping a
second copy of every panel in the repo, Figures/ holds symlinks into the run
directories that experiment_A and experiment_B write to, so a re-run of an
experiment updates the document with no copying.

Figure names carry the whole run configuration EXCEPT the feedback model --
that selects the directory, not the stem -- so the same name can exist under
both fiducial/ and strongest_AGN/. --feedback says which one these notes are
about; anything taken from another run is reported loudly, because a figure
from the wrong feedback model looks entirely plausible and is wrong.

The rest of the name does carry the configuration, so a name in the .tex that
nothing matches means that run has not been done -- not that a file went
astray. Those are reported as MISSING and main.tex draws a placeholder box in
their place, so the document always compiles.

    python paper_notes/link_figures.py          # relink, report
    python paper_notes/link_figures.py --copy   # real files, for Overleaf
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLOTS = ROOT / 'plots'
FIGDIR = HERE / 'Figures'
TEX = HERE / 'main.tex'


def referenced(tex):
    """Basenames of every figure main.tex asks for, in order of appearance."""
    src = tex.read_text()
    out = []
    for path in re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]*)\}', src):
        name = Path(path).name
        if name not in out:
            out.append(name)
    return out


DEFAULT_FEEDBACK = 'strongest_AGN'


def index(plots):
    """Every pdf under plots/, by basename. A name can occur in more than one
    run directory (fiducial and strongest_AGN produce the same stems when the
    knobs agree), so keep them all and let the caller decide."""
    found = {}
    for p in sorted(plots.rglob('*.pdf')):
        found.setdefault(p.name, []).append(p)
    return found


def pick(hits, feedback):
    """(chosen, from_wanted_feedback) for one stem.

    Sorting alone would pick fiducial over strongest_AGN, silently, for every
    stem that exists under both -- which is how six panels of a strongest_AGN
    document came to be the fiducial run.
    """
    wanted = [h for h in hits if f'/{feedback}/' in h.as_posix()]
    return (wanted[0], True) if wanted else (hits[0], False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--copy', action='store_true',
                    help="copy the files instead of linking them, for a "
                         "self-contained tree to upload to Overleaf")
    ap.add_argument('--feedback', default=DEFAULT_FEEDBACK,
                    help="which run directory these notes are about; a stem "
                         "present under more than one is taken from this one")
    args = ap.parse_args()

    if not TEX.exists():
        sys.exit(f"No {TEX}")
    FIGDIR.mkdir(exist_ok=True)

    # Clear only what this script put there, so a figure dropped in by hand
    # (one built outside the pipeline) survives a relink.
    for old in FIGDIR.iterdir():
        if old.is_symlink():
            old.unlink()

    have = index(PLOTS)
    linked, missing, ambiguous, wrong_run = [], [], [], []
    for name in referenced(TEX):
        hits = have.get(name, [])
        dest = FIGDIR / name
        if not hits:
            if dest.exists():          # dropped in by hand, not ours to judge
                linked.append((name, dest.relative_to(ROOT), 'kept'))
            else:
                missing.append(name)
            continue
        src, ours = pick(hits, args.feedback)
        if not ours:
            wrong_run.append((name, src))
        elif len(hits) > 1:
            ambiguous.append((name, hits, src))
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if args.copy:
            shutil.copy2(src, dest)
        else:
            dest.symlink_to(Path('..') / '..' / src.relative_to(ROOT))
        linked.append((name, src.relative_to(ROOT), 'copy' if args.copy else 'link'))

    verb = 'copied' if args.copy else 'linked'
    print(f"{len(linked)} {verb}, {len(missing)} missing "
          f"({len(referenced(TEX))} referenced by main.tex), "
          f"feedback = {args.feedback}\n")
    for name, src, how in linked:
        print(f"  [{how:4s}] {src}")
    if ambiguous:
        print(f"\nSame stem under more than one run; took {args.feedback}:")
        for name, hits, src in ambiguous:
            print(f"  {name}")
            for h in hits:
                mark = '  <-- used' if h == src else ''
                print(f"      {h.relative_to(ROOT)}{mark}")
    if wrong_run:
        print(f"\n*** {len(wrong_run)} figure(s) are NOT from {args.feedback} "
              f"and no {args.feedback} version exists. The document will show "
              f"another run\'s result under a name that looks right:")
        for name, src in wrong_run:
            print(f"      {src.relative_to(ROOT)}")
    if missing:
        print(f"\n{len(missing)} referenced by main.tex but not in plots/ "
              f"-- main.tex draws a placeholder box for each:")
        for name in missing:
            print(f"  {name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
