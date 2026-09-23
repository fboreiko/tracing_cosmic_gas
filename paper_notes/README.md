# paper_notes

`main.tex` — the HATF project notes. Builds to `main.pdf`.

```bash
cd paper_notes && make
```

`make` refreshes `Figures/` and then runs `latexmk -pdf` until the cross
references settle. Other targets:

| target | does |
|---|---|
| `make` | `main.pdf` |
| `make figures` | relink `Figures/` from `../plots` |
| `make watch` | rebuild on every save (`latexmk -pvc`) |
| `make overleaf` | replace the symlinks with real files, for a zip upload |
| `make clean` / `distclean` | drop aux files / aux files and the pdf |

TeX Live is already on this machine at `/Library/TeX/texbin`.

## Figures

`main.tex` writes `\includegraphics{Figures/<name>.pdf}`, and `Figures/` holds
**symlinks** into `../plots/pme_reconstruction/<feedback>/`, rebuilt by
`link_figures.py`. Re-running an experiment therefore updates the document with
no copying and no second copy of any panel in the repo. `Figures/` is not in
git — `make` regenerates it.

A figure's filename encodes its entire run configuration (feedback, binning,
apertures, extrapolation modes, weights, concentration source), so **a name
with nothing behind it means that run has not been done**, not that a file was
lost. `main.tex` draws a labelled box in place of each one instead of failing
the build, so the document always compiles while results are still coming in.

A figure built outside the pipeline can be dropped into `Figures/` by hand;
`link_figures.py` only removes symlinks, so it will survive a relink.

### The run these figures come from

Every figure is the **strongest_AGN** run measured with **log-spaced k bins**
(25 per decade, width floored at 2 k_f) and the large scales measured in 3-D on
a 384^3 grid, spliced under the projected measurement below k = 1:

```bash
python -m Pmx_reconstruction.experiment_A --ngrid-3d 384 --k-split 1.0
python -m Pmx_reconstruction.experiment_B --ngrid-3d 384 --k-split 1.0 \
        --apertures 1.0 1.2 1.5 2.0
```

which is why current names end in `_3d384k1`. The linear-binning figures they
replaced are kept under
`plots/pme_reconstruction/<feedback>/k_linspace_binning_legacy/`.

`PmxConfig`'s defaults supply the rest (`concentration_source = 'colossus'`,
so no `_conc-colossus` suffix). S_eff(k -> 0) = 0.944 on this run, against
0.936 on the projected measurement alone.

### Not built yet (1 of 8)

**Split validation panel** (Figure `split_validation_test_mg`) -- the only
figure `main.tex` asks for that this run did not produce, so it shows a
placeholder box:

```bash
python -m Pmx_reconstruction.experiment_A --validate --logm-r 12 \
        --hmf catalog --ngrid-3d 384 --k-split 1.0
```

The name in `main.tex` was also updated from the pre-restructure scheme
(`..._e_..._split12.00_ref12.00_...`) to what the current code writes.

### There is no way to select the feedback run from the command line

`PmxConfig.feedback` defaults to `'strongest_AGN'` and `base_parser` never
exposes it, so every entry point is hard-wired to that run. The fiducial
figures in `plots/` were produced by editing the default in
`pmxlib/config.py` by hand. Adding a `--feedback` argument to the simulation
group would remove a footgun: a run intended for one model silently lands in
the other model's plot directory under a name that looks right.

This has already bitten once. A fiducial run produced the same stems as the
strongest_AGN ones, and because `link_figures.py` indexed by basename and took
the first hit -- and `fiducial` sorts before `strongest_AGN` -- six of the
eight panels silently became the fiducial run while the text quoted
strongest_AGN numbers. `link_figures.py` now takes `--feedback`
(default `strongest_AGN`), prefers that directory for any stem found under
more than one, and reports loudly when a figure has no version under the
requested run. **The feedback model is the one part of the configuration the
filename does not carry**, so it is the one thing the stem cannot protect you
from.
