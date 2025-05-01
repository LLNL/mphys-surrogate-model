from PySDM_examples.Shipway_and_Hill_2012 import Settings, Simulation
from PySDM.physics import si
from PySDM.exporters import NetCDFExporter_1d
import numpy as np

common_params = {
    "n_sd_per_gridbox": 256,
    "dt": 5 * si.s,
    "dz": 50 * si.m,
    "p0": 990 * si.hPa,
    "kappa": 0.9,
    "save_spec_and_attr_times": np.linspace(0, 60 * si.minutes + 0, 361),
}

output = {}
settings = {}
simulation = {}
for rho_times_w in (
    2 * si.kg / si.m**3 * si.m / si.s,
    3 * si.kg / si.m**3 * si.m / si.s,
):
    for particles_per_volume_STP in (
        50,
        100,
        200,
    ):
        for case_iter in range(10):
            key = f"rhow={rho_times_w}_N={particles_per_volume_STP}_{case_iter}"
            print(key)
            settings[key] = Settings(
                **common_params,
                rho_times_w_1=rho_times_w,
                particles_per_volume_STP=particles_per_volume_STP / si.cm**3,
            )
            settings[key].r_bins_edges = np.logspace(
                np.log10(0.1 * si.um), np.log10(10 * si.mm), 101, endpoint=True
            )
            simulation[key] = Simulation(settings[key])
            output[key] = simulation[key].run().products
            nc_exporter = NetCDFExporter_1d(
                output[key],
                settings[key],
                simulation[key],
                "box_data_64/" + key + ".nc",
            )
            nc_exporter.run()
            print("saved")
