from qutip import basis, tensor, destroy, qeye, sigmax, sigmaz, mesolve, expect as q_expect
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N = 3
t_total = 10
tlist1 = np.linspace(0, t_total / 2, 250)
tlist2 = np.linspace(t_total / 2, t_total, 250)

# Modified parameters for better visibility
g = [0.0, 1.0, 0.0]      # Only qubit 1 is driven
J = [0.2, 0.2]           # Reduced crosstalk for more balanced effects
Delta = [0.0, 0.0, 0.0]  # No detuning

# --- Operators ---
sm_list, sz_list, sx_list = [], [], []

for i in range(N):
    op_list = [qeye(2) for _ in range(N)]
    op_list[i] = destroy(2)
    sm_list.append(tensor(op_list))

for i in range(N):
    op_list = [qeye(2) for _ in range(N)]
    op_list[i] = sigmaz()
    sz_list.append(tensor(op_list))

for i in range(N):
    op_list = [qeye(2) for _ in range(N)]
    op_list[i] = sigmax()
    sx_list.append(tensor(op_list))

# --- Hamiltonian ---
H = 0
for i in range(N):
    H += g[i] * (sm_list[i] + sm_list[i].dag())  # Rabi driving
    H += 0.5 * Delta[i] * sz_list[i]             # Detuning

# Nearest-neighbor crosstalk
H += J[0] * sx_list[0] * sx_list[1]  # 0-1 coupling
H += J[1] * sx_list[1] * sx_list[2]  # 1-2 coupling

# --- Initial state: all in |0⟩ ground state ---
psi0 = tensor([basis(2, 0) for _ in range(N)])

# --- Expectation operators: measure sigma_z ---
e_ops = sz_list

# --- First evolution (before π pulse) ---
# Force state saving with explicit options
from qutip import Options
opts = Options(store_states=True)
result1 = mesolve(H, psi0, tlist1, [], [], options=opts)
psi_after_first_half = result1.states[-1]

print("State populations after first half:")
for i in range(N):
    pop_excited = abs(psi_after_first_half.ptrace(i)[1,1])**2
    print(f"Qubit {i}: P(|1⟩) = {pop_excited:.4f}")

# --- Central π pulse on qubit 1 ---
U_pi_center = (-1j * np.pi * sx_list[1] / 2).expm()
psi_after_center = U_pi_center * psi_after_first_half

# --- Echo π pulses on qubits 0 and 2 (applied simultaneously) ---
U_echo_0 = (-1j * np.pi * sx_list[0] / 2).expm()
U_echo_2 = (-1j * np.pi * sx_list[2] / 2).expm()
# Apply them in sequence to avoid issues with non-commuting operators
psi_after_echo = U_echo_2 * U_echo_0 * psi_after_center

print("\nState populations after all pulses:")
for i in range(N):
    pop_excited = abs(psi_after_echo.ptrace(i)[1,1])**2
    print(f"Qubit {i}: P(|1⟩) = {pop_excited:.4f}")

# --- Second evolution ---
result2 = mesolve(H, psi_after_echo, tlist2, [], e_ops, options=opts)

# --- Combine results ---
tlist = np.concatenate([tlist1, tlist2])

# For the first evolution, we need to manually compute expectations since we didn't pass e_ops
expect1 = []
for i in range(N):
    expect_vals = [q_expect(sz_list[i], state) for state in result1.states]
    expect1.append(expect_vals)

expect_values = []
for i in range(N):
    # Combine expectation values from both evolution periods
    expect_combined = np.concatenate([expect1[i], result2.expect[i]])
    expect_values.append(expect_combined)

# --- Plotting with improved visibility ---
labels = ['Qubit 0 ⟨σ_z⟩', 'Qubit 1 ⟨σ_z⟩', 'Qubit 2 ⟨σ_z⟩']
colors = ['blue', 'red', 'green']
linestyles = ['-', '-', '-']

plt.figure(figsize=(12, 8))

for i, exp in enumerate(expect_values):
    plt.plot(tlist, exp, label=labels[i], color=colors[i], 
             linestyle=linestyles[i], linewidth=2)

# Mark the pulse location
plt.axvline(tlist1[-1], color='black', linestyle='--', alpha=0.7, 
            label='π pulses applied', linewidth=2)

plt.xlabel('Time', fontsize=12)
plt.ylabel('Expectation ⟨σ_z⟩', fontsize=12)
plt.title('Dynamical Decoupling with Crosstalk: Fixed Version', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add text box with information
textstr = f'Driving: Qubit 1 only (g={g[1]})\nCrosstalk: J₀₁={J[0]}, J₁₂={J[1]}\nPulses at t={t_total/2:.1f}: π on all qubits'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.ylim(-1.1, 1.1)  # Set y-axis limits to show full range
plt.show()

# --- Print final expectation values for verification ---
print(f"\nFinal ⟨σ_z⟩ values at t={t_total}:")
for i in range(N):
    print(f"Qubit {i}: {expect_values[i][-1]:.4f}")