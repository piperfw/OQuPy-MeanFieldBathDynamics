#!/usr/bin/env python
import sys
sys.path.insert(0,'..')
import oqupy
import numpy as np
import matplotlib.pyplot as plt
from oqupy import system_dynamics
from oqupy import bath_dynamics

alpha = 0.2
nuc = 0.15
T = 0.026
Omega = 0.3
omega0_1, omega0_2 = 0.0, 0.4
omegac = 0.0
kappa = 0.01
Gamma_down = 0.01
Gamma_up = 0.8 * Gamma_down

sigma_z = oqupy.operators.sigma("z")
sigma_plus = oqupy.operators.sigma("+")
sigma_minus = oqupy.operators.sigma("-")

def H_MF_1(t, a):
    return 0.5 * omega0_1 * sigma_z +\
        0.5 * Omega * (a * sigma_plus + np.conj(a) * sigma_minus)
def H_MF_2(t, a):
    return 0.5 * omega0_2 * sigma_z +\
        0.5 * Omega * (a * sigma_plus + np.conj(a) * sigma_minus)

fractions = [0.5, 0.5]
def field_eom(t, states, field):
    sx_exp_list = [np.matmul(sigma_minus, state).trace() for state in states]
    sx_exp_weighted_sum = sum([fraction*sx_exp for fraction, sx_exp in zip(fractions, sx_exp_list)])
    return -(1j*omegac+kappa)*field - 0.5j*Omega*sx_exp_weighted_sum


subsystem_1 = oqupy.TimeDependentSystemWithField(H_MF_1)
subsystem_2 = oqupy.TimeDependentSystemWithField(H_MF_2)
correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=nuc,
                                cutoff_type='gaussian',
                                temperature=T)
bath = oqupy.Bath(0.5 * sigma_z, correlations)
initial_field = np.sqrt(0.05)
initial_state_1 = np.array([[0,0],[0,1]])
initial_state_2 = np.array([[0,0],[0,1]])
initial_state_list = [initial_state_1, initial_state_2]

tempo_parameters = oqupy.TempoParameters(dt=0.2, tcut=2.0, epsrel=10**(-4))
start_time = 0.0
end_time = 20

mean_field_system = oqupy.MeanFieldSystem([subsystem_1, subsystem_2], field_eom=field_eom)

# SYSTEM DYNAMICS
process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=start_time,
                                        end_time=end_time,
                                        parameters=tempo_parameters)
process_tensor_list = [process_tensor, process_tensor]

control_list = [oqupy.Control(subsystem_1.dimension), oqupy.Control(subsystem_2.dimension)]

mean_field_dynamics_process = \
        system_dynamics.compute_dynamics_with_field(mean_field_system,
                initial_field=initial_field, 
                initial_state_list=initial_state_list, 
                start_time=start_time,
                process_tensor_list = process_tensor_list)

# BATH DYNAMICS
bath_corr = oqupy.bath_dynamics.TwoTimeBathCorrelations(mean_field_system,
                                                        bath, 
                                                        process_tensor_list,
                                                        initial_state_list,
                                                        initial_field=initial_field,
                                                        target_mean_field_system=0) # WHICH SPECIES (0 or 1) to get environment dynamics for
w = Omega # Frequency of bath mode to probe
delta = 0.1 * Omega # Bandwidth in frequency to probe 
tlist, occ = bath_corr.occupation(w, delta, change_only = True) # occupation
energy = w * occ # energy 

# BATH DYNAMICS 2
bath_corr2 = oqupy.bath_dynamics.TwoTimeBathCorrelations(mean_field_system,
                                                        bath, 
                                                        process_tensor_list,
                                                        initial_state_list,
                                                        initial_field=initial_field,
                                                        target_mean_field_system=1) # CHANGED
tlist2, occ2 = bath_corr2.occupation(w, delta, change_only = True) # occupation
energy2 = w * occ2 # energy 


fig, axes = plt.subplots(2, figsize=(9,6), sharex=True)
times_pt, fields_pt = mean_field_dynamics_process.field_expectations()
axes[0].plot(times_pt, np.abs(fields_pt)**2)
axes[0].set_ylabel('$n/N$', rotation=0, labelpad=20)
axes[1].plot(tlist, occ, label='Bath 1')
axes[1].plot(tlist2, occ2, label='Bath 2')
axes[1].legend()
axes[1].set_xlabel(r'$t$')
axes[1].set_ylabel(r'$\Delta Q ( \Omega, t)$', rotation=0, labelpad=30)
axes[0].set_title(r'Two-species bath dynamics')

fig.savefig('bath_dynamics.png', bbox_inches='tight', dpi=350)
