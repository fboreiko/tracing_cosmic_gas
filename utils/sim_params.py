"""Per-simulation parameters.

FLAMINGO's numbers are literals and cost nothing. Abacus's are read from
abacusnbody's metadata database, which is a real dependency: it has to be
installed, and it has to be able to resolve AbacusSummit_base_c000_ph000.

That lookup used to run at module scope, so importing anything anywhere in the
repo required it -- including the Pmx reconstruction, which never touches
Abacus and whose FLAMINGO parameters are the literals below. On a machine
without abacusnbody, or offline, that made the whole pipeline unimportable for
no reason.

It is now deferred to the first request for sim='abacus' and cached, so
get_sim_params('flamingo') needs nothing beyond the standard library.
"""
from functools import lru_cache

_FLAMINGO = {
    "name": "flamingo",
    "label": "FLAMINGO",
    "box_size_cMpc_h": 681.0,
    "redshift": 0.74,
    "ngrid_default": 2048,
    "nthread_default": 4,
    "h": 0.681,
    "omega_m": 0.306,
    "omega_b": 0.0486,
    "sigma8": 0.807,
    "n_s": 0.967,
    "tcmb0": 2.725,
    "X_H": 0.76,
    "X_He": 0.24,
}

# Names this module knows about, without needing to resolve any of them. Kept
# separate from the built parameter dicts so that listing the valid options in
# an error message never triggers the Abacus lookup.
KNOWN_SIMS = ("flamingo", "abacus")

ABACUS_SIM_NAME = "AbacusSummit_base_c000_ph000"
ABACUS_REDSHIFT = 0.8


@lru_cache(maxsize=1)
def abacus_meta():
    """Raw abacusnbody metadata for the fiducial AbacusSummit box.

    Imported and looked up here rather than at module scope. Cached, so the
    cost is paid once per process and only if something asks for Abacus.
    """
    try:
        from abacusnbody.metadata import get_meta
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            "sim='abacus' needs the abacusnbody package, which is not "
            "installed. FLAMINGO does not need it: its parameters are "
            "literals in utils/sim_params.py."
        ) from exc
    return get_meta(ABACUS_SIM_NAME, redshift=ABACUS_REDSHIFT)


@lru_cache(maxsize=1)
def _abacus_params():
    """Abacus parameters, derived from the metadata on first use."""
    meta = abacus_meta()
    h = meta["H0"] / 100.0
    omega_b = meta["omega_b"] / (h ** 2)
    omega_cdm = meta["omega_cdm"] / (h ** 2)
    box_size = float(meta.get("BoxSizeHMpc", meta["BoxSize"]))
    return {
        "name": "abacus",
        "label": "Abacus",
        "box_size_cMpc_h": box_size,
        "redshift": ABACUS_REDSHIFT,
        "ngrid_default": 6144,
        "nthread_default": 4,
        "h": h,
        "omega_m": omega_cdm + omega_b,
        "omega_b": omega_b,
        "sigma8": 0.807952,
        "n_s": 0.9649,
        "tcmb0": 2.725,
        "X_H": 0.76,
        "X_He": 0.24,
    }


def get_sim_params(sim: str = "flamingo") -> dict:
    """Parameters for one simulation, as a fresh dict the caller may mutate.

    Only resolves what is asked for: requesting 'flamingo' never touches
    abacusnbody.
    """
    sim = (sim or "flamingo").lower()
    if sim == "flamingo":
        return _FLAMINGO.copy()
    if sim == "abacus":
        return _abacus_params().copy()
    raise ValueError(f"Unknown sim '{sim}'. Valid options: {sorted(KNOWN_SIMS)}")


def require_sim_param(sim: str, key: str):
    params = get_sim_params(sim)
    value = params.get(key)
    if value is None:
        raise NotImplementedError(
            f"Simulation parameter '{key}' is not yet defined for sim='{sim}'."
        )
    return value


if __name__ == "__main__":
    sim_name = "abacus"
    params = get_sim_params(sim_name)
    print(params["h"])
    print(params["omega_b"])
    print(params["omega_m"] + params["omega_b"])

    # Extract and print ParticleMassHMsun header field
    particle_mass_msun_h = abacus_meta().get('ParticleMassHMsun', None)
    if particle_mass_msun_h is not None:
        particle_mass_1e10_msun_h = particle_mass_msun_h / 1e10
        print(f"\nParticleMassHMsun: {particle_mass_msun_h} msun/h")
        print(f"ParticleMassHMsun: {particle_mass_1e10_msun_h} 10^10 msun/h")
    else:
        print(f"\nParticleMassHMsun: Not found")