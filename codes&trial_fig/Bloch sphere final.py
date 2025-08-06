import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- 1. RUN THE 3-QUBIT SIMULATION (same as before) ---
N = 3
target = 1
w_q   = 5.0 * 2*np.pi    # 5 GHz → rad/ns
Omega = 10.0 * 2*np.pi   # 10 GHz → rad/ns
J     = 0.1 * Omega

t_total = 100.0
t0, tau = 50.0, 5.0
def pulse(t, args):
    return np.exp(- (t-args['t0'])**2/(2*args['tau']**2))

T1, T2 = 50e3, 10e3
γ1, γ2 = 1/T1, 1/T2

# build Pauli operators for each qubit
sx = [qt.tensor(*( [qt.qeye(2)]*i + [qt.sigmax()] + [qt.qeye(2)]*(N-1-i) )) for i in range(N)]
sy = [qt.tensor(*( [qt.qeye(2)]*i + [qt.sigmay()] + [qt.qeye(2)]*(N-1-i) )) for i in range(N)]
sz = [qt.tensor(*( [qt.qeye(2)]*i + [qt.sigmaz()] + [qt.qeye(2)]*(N-1-i) )) for i in range(N)]
sm = [qt.tensor(*( [qt.qeye(2)]*i + [qt.sigmam()] + [qt.qeye(2)]*(N-1-i) )) for i in range(N)]

H_drift = sum(-0.5*w_q*sz[i] for i in range(N))
H_drive =  Omega * sx[target]
H_xtalk =  J * (sx[target-1]*sx[target] + sx[target]*sx[target+1])
H = [H_drift, [H_drive+H_xtalk, pulse]]

c_ops = []
for i in range(N):
    c_ops += [np.sqrt(γ1)*sm[i], np.sqrt(γ2)*sz[i]]

psi0  = qt.tensor(*[qt.basis(2,0) for _ in range(N)])
tlist = np.linspace(0, t_total, 201)
args  = {'t0': t0, 'tau': tau}

print("Running simulation...")
result = qt.mesolve(H, psi0, tlist, c_ops, [], args=args)
print("Simulation complete.")

# --- 2. COMPUTE BLOCH VECTOR COMPONENTS ---
expect_x = [qt.expect(sx[i], result.states) for i in range(N)]
expect_y = [qt.expect(sy[i], result.states) for i in range(N)]
expect_z = [qt.expect(sz[i], result.states) for i in range(N)]

# --- 3. SET UP THREE BLOCH SPHERES AND ANIMATE THEM ---
fig = plt.figure(figsize=(15, 5))
axes = []
colors = ['green', 'saddlebrown', 'royalblue']
labels = [r'$\psi_0$', r'$\psi_1$', r'$\psi_2$']

# create 3 subplots, one sphere per qubit
for i in range(N):
    ax = fig.add_subplot(1, N, i+1, projection='3d')
    ax.set_box_aspect((1,1,1))
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"Qubit {i}")
    # draw wireframe sphere
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0,   np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.3)
    axes.append(ax)

# containers for arrows and labels
arrow_artists = [[] for _ in range(N)]
text_artists  = [[] for _ in range(N)]

def init():
    return []

def update(frame):
    # remove previous arrows/text
    for i in range(N):
        for art in arrow_artists[i] + text_artists[i]:
            art.remove()
        arrow_artists[i].clear()
        text_artists[i].clear()
    # draw new arrows
    for i, ax in enumerate(axes):
        x, y, z = expect_x[i][frame], expect_y[i][frame], expect_z[i][frame]
        # draw a thick arrow from origin
        art = ax.quiver(
            0,0,0,    # origin
            x,y,z,    # vector
            color=colors[i],
            linewidth=3,
            arrow_length_ratio=0.2
        )
        arrow_artists[i].append(art)
        # label just beyond tip
        txt = ax.text(
            1.1*x, 1.1*y, 1.1*z,
            labels[i],
            color=colors[i],
            fontsize=14,
            weight='bold'
        )
        text_artists[i].append(txt)
    # return all artists so matplotlib knows what to redraw
    all_arts = sum(arrow_artists, []) + sum(text_artists, [])
    return all_arts

ani = FuncAnimation(
    fig, update,
    frames=len(tlist),
    init_func=init,
    blit=False,       # disable blit so arrows show reliably
    interval=50
)

# save and display
ani.save('bloch_three_vectors.gif', writer='pillow', fps=20)
plt.show()
