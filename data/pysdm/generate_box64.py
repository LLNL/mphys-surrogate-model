#%%
import numpy as np
import warnings
from PySDM import Formulae
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.physics import si, terminal_velocity
from PySDM.initialisation import spectra
from PySDM.dynamics.collisions.collision_kernels import Hydrodynamic
from PySDM.initialisation.sampling.spectral_sampling import UniformRandom
from PySDM.products import ParticleVolumeVersusRadiusLogarithmSpectrum
from scipy.stats import qmc
#%%
class Settings:
    def __init__(
        self,
        *,
        seed: int = 0,
        n_sd: int = 2**13,
        n_part: float = 2**23 / si.metre**3,
        dt: float = 1 * si.s,
        steps=None,
        kernel=None,
        dv: float = 1e6 * si.metres**3,
        spectrum=None,
        radius_bins_edges = np.logspace(
            np.log10(1 * si.um), np.log10(5e3 * si.um), num=65, endpoint=True
        )            
    ):  
        self.formulae = Formulae(seed=seed, terminal_velocity='RogersYau')
        self.n_sd = n_sd
        self.n_part = n_part
        self.dv = dv
        self.norm_factor = self.n_part * self.dv 
        self.rho = 1000 * si.kilogram / si.metre**3
        self.dt = dt
        self.adaptive=False
        self.steps = steps
        self.kernel = kernel
        self.spectrum = spectrum
        self.radius_bins_edges = radius_bins_edges

    @property
    def output_steps(self):
        return [int(step / self.dt) for step in self.steps]

#%%
class Simulation:
    def __init__(self, settings, backend=CPU):
        self.nt = int(settings.output_steps[-1] / settings.dt)
        self.save_spec_and_attr_times = settings.output_steps
        self.number_of_bins = len(settings.radius_bins_edges)

        self.particulator = None
        self.output_attributes = None
        self.output_products = None

        env = Box(dv=settings.dv, dt=settings.dt)
        self.backend = backend(formulae=settings.formulae)
        self.builder = Builder(n_sd=settings.n_sd, backend=self.backend, environment=env)
        
        attributes = {}
        sampling = UniformRandom(settings.spectrum) #Logarithmic(settings.spectrum)
        attributes["volume"], attributes["multiplicity"] = sampling.sample(settings.n_sd, backend=self.backend)
        self.attributes = attributes

        coalescence = Coalescence(
            collision_kernel=settings.kernel, adaptive=settings.adaptive
        )
        self.builder.add_dynamic(coalescence)

        self.products = (
            ParticleVolumeVersusRadiusLogarithmSpectrum(
                settings.radius_bins_edges, name="dv/dlnr"
            ),
        )

        self.particulator = self.builder.build(
            attributes=self.attributes, products=self.products
        )

        self.output_products = {}
        for k, v in self.particulator.products.items():
            if len(v.shape) == 0:
                self.output_products[k] = np.zeros(self.nt + 1)
            elif len(v.shape) == 1:
                self.output_products[k] = np.zeros((self.nt + 1))
            elif len(v.shape) == 2:
                number_of_time_sections = len(self.save_spec_and_attr_times)
                self.output_products[k] = np.zeros(
                    (self.number_of_bins - 1, number_of_time_sections)
                )

    def save_products(self, index):
        for k, v in self.particulator.products.items():
            if len(v.shape) == 1:
                self.output_products[k][:, index] = v.get()
            elif len(v.shape) == 2:
                self.output_products[k][:, index] = v.get()[0]

    def run(self):
        assert "t" not in self.output_products
        self.output_products["t"] = self.save_spec_and_attr_times

        self.save_products(0)
        for (idx, step) in enumerate(self.save_spec_and_attr_times):
            self.particulator.run(steps=self.nt)
            self.save_products(idx)

        Outputs = namedtuple("Outputs", "products attributes")
        output_results = Outputs(self.output_products, self.output_attributes)
        return output_results

#%%
"""
netcdf exporter of scalar products for 1d simulations
"""

from collections import namedtuple
from copy import deepcopy

import numpy as np
from scipy.io import netcdf_file


class NetCDFExporter_0d:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    def __init__(  # pylint: disable = too-many-arguments
        self, data, settings, simulator, filename,
    ):
        self.data = data
        self.settings = settings
        self.simulator = simulator
        self.vars = None
        self.filename = filename
        self.n_save_spec = len(self.settings.output_steps)

    def _create_dimensions(self, ncdf):
        ncdf.createDimension("time", self.n_save_spec)
        ncdf.createDimension("height", 1)

        for name, instance in self.simulator.particulator.products.items():
            if len(instance.shape) == 2:
                dim_name = name.replace(" ", "_") + "_bin_index"
                ncdf.createDimension(dim_name, self.simulator.number_of_bins - 1)

    def _create_variables(self, ncdf):
        self.vars = {}

        self.vars["time"] = ncdf.createVariable(
            "time", "f", ["time"]
        )
        self.vars["time"][:] = self.settings.output_steps
        self.vars["time"].units = "seconds"
        

        self.vars["height"] = ncdf.createVariable("height", "f", ["height"])
        self.vars["height"][:] = [0.0]
        self.vars["height"].units = "metres"

        for name, instance in self.simulator.particulator.products.items():
            if len(instance.shape) == 2:
                label = name.replace(" ", "_") + "_bin_index"
                self.vars[label] = ncdf.createVariable(label, "f", (label,))
                self.vars[label][:] = self.settings.radius_bins_edges[1:]

        for name, instance in self.simulator.particulator.products.items():
            if name in self.vars:
                raise AssertionError(
                    f"product ({name}) has same name as one of netCDF dimensions"
                )

            n_dimensions = len(instance.shape)
            if n_dimensions == 0:
                dimensions = ("time",)
            elif n_dimensions == 1:
                dimensions = ("height", "time")
            elif n_dimensions == 2:
                dim_name = name.replace(" ", "_") + "_bin_index"
                if self.n_save_spec == 0:
                    continue
                if self.n_save_spec == 1:
                    dimensions = (f"{dim_name}")
                else:
                    dimensions = (f"{dim_name}", "time")
            else:
                raise NotImplementedError()

            self.vars[name] = ncdf.createVariable(name, "f", dimensions)
            self.vars[name].units = instance.unit

    def _write_variables(self):
        for var in self.simulator.particulator.products.keys():
            n_dimensions = len(self.simulator.particulator.products[var].shape)
            if n_dimensions == 0:
                self.vars[var][:] = self.data[var][:]
            elif n_dimensions == 1:
                self.vars[var][:] = self.data[var][:]
            elif n_dimensions == 2:
                self.vars[var][:, :] = self.data[var][:, :]
            else:
                raise NotImplementedError()

    def run(self):
        with netcdf_file(self.filename, mode="w") as ncdf:
            self._create_dimensions(ncdf)
            self._create_variables(ncdf)
            self._write_variables()
            ncdf.close()

#%%
T_end = 2 #minutes
dt_out = 10 # seconds
common_params = {
    "n_sd": 2**11,
    "dt": 1 * si.s,
    "steps": np.linspace(0, T_end*si.minutes, int(T_end*si.minutes/dt_out) + 1).tolist(),
    "kernel": Hydrodynamic(),
    "dv": 100 * si.metres**3
}
#%%
n_runs = 1
n_samples = 3
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=n_samples)

Nd_per_cm3 = 10**(0 + 3 * sample[:, 0]) #(10, 20, 50, 100, 200, 400)
q_g_per_kg = 10**(-2 + 3.0 * sample[:, 1]) #(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.6, 2.0)
ks = 1 + 3.0 * sample[:, 2] #(1, 1.5, 2, 2.5, 3, 3.5, 4)
dry_air_density = 1 * si.kg / si.m**3
rhow = 1000 * si.kg / si.m**3

coal_tol = 1e-7

settings = {}
output = {}
simulation = {}
keys = []

warnings.filterwarnings("ignore", category=UserWarning) 
for i in range(n_samples):
    Nd = Nd_per_cm3[i]
    q = q_g_per_kg[i]
    k = ks[i]
    print(f"Nd={Nd}cm-3_q={q}gkg_k={k}")
    for ir in range(n_runs):
        mean_particle_mass = q * si.g / si.kg * dry_air_density / (Nd / si.cm**3)
        mean_particle_volume = mean_particle_mass / rhow
        norm_factor = Nd / si.cm**3 * common_params["dv"]
        theta = mean_particle_volume / k
        spectrum = spectra.Gamma(norm_factor=norm_factor, k = k, theta = theta)

        key = f"Nd={Nd}cm-3_q={q}gkg_k={k}_{ir}"
        keys.append(key)

        seed = int(Nd * 10 + q * 10 + k) + ir*100
        settings[key] = Settings(
            **common_params,
            spectrum=spectrum,
            seed = seed
        )
        simulation[key] = Simulation(settings[key])
        output[key] = simulation[key].run().products
        nc_exporter = NetCDFExporter_0d(output[key], settings[key], simulation[key], "box_data_64/" + key + ".nc")
        nc_exporter.run()

        coal_diff = np.linalg.norm(output[key]['dv/dlnr'][:,-1] - output[key]['dv/dlnr'][:,0])
        if coal_diff < coal_tol:
            print(f"{ir+1} runs; Not enough coalescence")
            break
    print(f"runs saved")