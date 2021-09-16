import os
import sys

import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["goose", "goose-latex"])


def rotation(theta):
    """
    Rotation matrix, based on angle.
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )


# simple shear strain
simple = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
    ]
)

# initialize strain measures
theta = np.linspace(-np.pi / 2.0, +np.pi / 2, 800)
eps_ps = np.empty(theta.shape)
eps_ss = np.empty(theta.shape)
eps_eq = np.empty(theta.shape)

# compute
for i in range(theta.size):

    R = rotation(theta[i])
    eps = (R.dot(simple)).dot(R.T)
    eps_ps[i] = eps[0, 0]
    eps_ss[i] = eps[0, 1]
    eps_eq[i] = np.sqrt(
        1.0 / 2.0 * (eps[0, 0] ** 2.0 + eps[1, 1] ** 2.0 + 2.0 * eps[0, 1] ** 2.0)
    )

# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots()

ax.plot(theta, eps_eq, label=r"$\varepsilon^\prime_\mathrm{eq}$")
ax.plot(theta, eps_ss, label=r"$\varepsilon^\prime_\mathrm{xx}$")
ax.plot(theta, eps_ps, label=r"$\varepsilon^\prime_\mathrm{xy}$")

plt.legend()

ax.xaxis.set_ticklabels([r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"])
ax.xaxis.set_ticks([-np.pi / 2.0, -np.pi / 4.0, 0, np.pi / 4.0, np.pi / 2.0])
ax.yaxis.set_ticks([-1, 0, 1])

plt.xlabel(r"$\theta$")
plt.ylabel(r"$\varepsilon / | \gamma |$")

plt.savefig(os.path.splitext(sys.argv[0])[0] + ".svg")
