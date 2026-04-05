from abacusnbody.metadata import get_meta

meta = get_meta("AbacusSummit_base_c000_ph000", redshift=0.8)

h = meta["H0"] / 100.0
omega_b = meta["omega_b"] / (h**2)
omega_cdm = meta["omega_cdm"] / (h**2) 
box_size = float(meta.get("BoxSizeHMpc", meta["BoxSize"]))

SIM_PARAMS = {
    "flamingo": {
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
    },
    "abacus": {
        "name": "abacus",
        "label": "Abacus",
        "box_size_cMpc_h": box_size,
        "redshift": 0.8,
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
    },
}


def get_sim_params(sim: str = "flamingo") -> dict:
    sim = (sim or "flamingo").lower()
    if sim not in SIM_PARAMS:
        raise ValueError(f"Unknown sim '{sim}'. Valid options: {sorted(SIM_PARAMS)}")
    return SIM_PARAMS[sim].copy()


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
    particle_mass_msun_h = meta.get('ParticleMassHMsun', None)
    if particle_mass_msun_h is not None:
        particle_mass_1e10_msun_h = particle_mass_msun_h / 1e10
        print(f"\nParticleMassHMsun: {particle_mass_msun_h} msun/h")
        print(f"ParticleMassHMsun: {particle_mass_1e10_msun_h} 10^10 msun/h")
    else:
        print(f"\nParticleMassHMsun: Not found")