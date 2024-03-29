import GMatElastoPlasticQPot as gm
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["goose", "goose-latex", "goose-tick-lower"])

# --------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 12))
fig.set_tight_layout(True)

# --------------------------------------------------------------------------------------------------

mat = gm.Cartesian2d.Cusp(1.0, 1.0, [-1.0, 1.0, 1.5, 3.0, 6.0, 10.1], False)

Eps = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
    ]
)

ninc = 20000
eps_xy = np.zeros(ninc)
sig_xy = np.zeros(ninc)
energy = np.zeros(ninc)

for i, d in enumerate(np.linspace(0, 10.0, ninc)):

    eps = d * Eps
    energy[i] = mat.energy(eps)
    sig = mat.Sig(eps)
    sig_xy[i] = sig[0, 1]
    eps_xy[i] = eps[0, 1]

ax = fig.add_subplot(2, 2, 1)

ax.plot(eps_xy, sig_xy)

plt.xlabel(r"$\varepsilon_\mathrm{xy}$")
plt.ylabel(r"$\sigma_\mathrm{xy}$")

ax = fig.add_subplot(2, 2, 2)

ax.plot(eps_xy, energy)

plt.xlabel(r"$\varepsilon_\mathrm{xy}$")
plt.ylabel(r"$E$")

# --------------------------------------------------------------------------------------------------

mat = gm.Cartesian2d.Smooth(1.0, 1.0, [-1.0, 1.0, 1.5, 3.0, 6.0, 10.1], False)

Eps = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
    ]
)

ninc = 20000
eps_xy = np.zeros(ninc)
sig_xy = np.zeros(ninc)
energy = np.zeros(ninc)

for i, d in enumerate(np.linspace(0, 10.0, ninc)):

    eps = d * Eps
    energy[i] = mat.energy(eps)
    sig = mat.Sig(eps)
    sig_xy[i] = sig[0, 1]
    eps_xy[i] = eps[0, 1]

ax = fig.add_subplot(2, 2, 3)

ax.plot(eps_xy, sig_xy)

plt.xlabel(r"$\varepsilon_\mathrm{xy}$")
plt.ylabel(r"$\sigma_\mathrm{xy}$")

ax = fig.add_subplot(2, 2, 4)

ax.plot(eps_xy, energy)

plt.xlabel(r"$\varepsilon_\mathrm{xy}$")
plt.ylabel(r"$E$")

# --------------------------------------------------------------------------------------------------

plt.savefig("stress-strain.svg")
plt.show()
