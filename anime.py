import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. SETUP THE SIMULATION ---

# System parameters (scaled for a 4-second animation)
N = 3
qubit_c = 1
w_q = 5.0 * 2 * np.pi        # Qubit frequency in rad/s (5 Hz)
Omega_c = 15.8 * 2 * np.pi   # Rabi frequency for a pi-pulse with this shape
J_coupling = 0.1 * Omega_c   # Induced coupling strength (10% of drive)

# Pulse parameters
t_total = 4.0
t0 = 1.0
tau = 0.1

def pulse_shape(t, args):
    """Gaussian pulse envelope."""
    return np.exp(-(t - args['t0'])**2 / (2 * args['tau']**2))

# Decoherence parameters
T1 = 10.0  # 10s T1 relaxation time
T2_star = 5.0  # 5s T2* dephasing time

#isolated system with no decoherence
gamma1 = 0 #1.0 / T1
gamma2_star = 0 #1.0 / T2_star

# --- 2. CONSTRUCT HAMILTONIAN AND OPERATORS (using Model B) ---

# Pauli operators for each qubit
sx = [qt.tensor([qt.qeye(2)]*i + [qt.sigmax()] + [qt.qeye(2)]*(N-1-i)) for i in range(N)]
sz = [qt.tensor([qt.qeye(2)]*i + [qt.sigmaz()] + [qt.qeye(2)]*(N-1-i)) for i in range(N)]
sm = [qt.tensor([qt.qeye(2)]*i + [qt.sigmam()] + [qt.qeye(2)]*(N-1-i)) for i in range(N)]

# Drift Hamiltonian
H_drift = -0.5 * w_q * sz[0] - 0.5 * w_q * sz[1] - 0.5 * w_q * sz[2]

# Control Hamiltonian operator
H_control_op = Omega_c * sx[qubit_c]

# Crosstalk Hamiltonian operator (Induced Coupling)
H_crosstalk_op = J_coupling * (sx[qubit_c-1] * sx[qubit_c] + sx[qubit_c] * sx[qubit_c+1])

# Full time-dependent Hamiltonian
H = [H_drift, [H_control_op + H_crosstalk_op, pulse_shape]]

# Collapse operators for Lindblad master equation
c_ops = []
for i in range(N):
    c_ops.append(np.sqrt(gamma1) * sm[i])
    c_ops.append(np.sqrt(gamma2_star) * sz[i])

# --- 3. RUN THE SIMULATION ---

# Initial state: all qubits in |0>
psi0 = qt.tensor(qt.basis(2,0), qt.basis(2,0), qt.basis(2,0))

# Time points for simulation (201 frames for a smooth GIF)
tlist = np.linspace(0, t_total, 201)
args = {'t0': t0, 'tau': tau}

print("Running QuTiP simulation...")
result = qt.mesolve(H, psi0, tlist, c_ops, e_ops=[sz[0], sz[1], sz[2]], args=args)
print("Simulation complete.")

# Extract expectation values
times = result.times
exp_sz0 = result.expect[0]
exp_sz1 = result.expect[1]
exp_sz2 = result.expect[2]

# --- 4. CREATE THE GIF ---

# Setup the plot
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, t_total)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel(r'$\langle \sigma_z \rangle$', fontsize=14)
ax.set_title('Qubit Evolution with Crosstalk', fontsize=16)
ax.grid(True)

# Plot the pulse shape in the background for context
# THIS IS THE CORRECTED LINE:
pulse_vals = [pulse_shape(t, args) for t in times]
ax.fill_between(times, -1.1, 1.1, where=np.array(pulse_vals) > 0.01, color='gray', alpha=0.2, label='Pulse Active')

# Initialize lines for each qubit and a time marker
line0, = ax.plot([], [], lw=3, color='yellow', linestyle='-', label='Neighbor (Qubit 0)')
line1, = ax.plot([], [], lw=2.5, color='red', label='Central (Qubit 1)')
line2, = ax.plot([], [], lw=1, color='blue', linestyle='-.', label='Neighbor (Qubit 2)')
time_marker = ax.axvline(0, color='black', lw=2, linestyle=':')
ax.legend(loc='lower left')

# Initialization function for the animation
def init():
    line0.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    time_marker.set_xdata([0])
    return line0, line1, line2, time_marker

# Update function called for each frame
def update(frame):
    # Update the data for each line up to the current frame
    line0.set_data(times[:frame+1], exp_sz0[:frame+1])
    line1.set_data(times[:frame+1], exp_sz1[:frame+1])
    line2.set_data(times[:frame+1], exp_sz2[:frame+1])
    
    # Update the time marker
    time_marker.set_xdata([times[frame]])
    
    return line0, line1, line2, time_marker

# Create the animation
ani = FuncAnimation(fig, update, frames=len(times),
                    init_func=init, blit=True)

# Save the animation as a GIF
gif_filename = 'qubit_crosstalk_evolution.gif'
print(f"Saving animation to {gif_filename}... (This may take a moment)")
ani.save(gif_filename, writer='pillow', fps=25)
print("GIF saved successfully.")

# Optional: show the final plot
plt.show()
