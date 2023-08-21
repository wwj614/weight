import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn import mixture

vals = np.loadtxt('./c.txt', usecols=0, delimiter='\t', unpack=True)
vals2 = np.array(list(zip(vals[1:], vals)))

x = np.linspace(83, 93, 101, endpoint=True)
y = np.linspace(83, 93, 101, endpoint=True)

g2 = mixture.GaussianMixture(n_components=2, covariance_type="full", tol=1e-3, max_iter=1000, random_state=103,
                             means_init=np.array([[86.29, 86.26], [89.08, 89.17]]),
                             weights_init=np.array([0.6549, 0.3451]))
g2.fit(vals2)
print(g2.weights_)
print(g2.means_)
print(g2.covariances_)

# Plot the contour.
X, Y = np.meshgrid(x, y)
XY = np.array([X.ravel(), Y.ravel()]).T
Z = np.exp(g2.score_samples(XY)) * 1000
Z = Z.reshape(X.shape)

CS = plt.contour(
    X, Y, Z, levels=[1, 2, 5, 10, 20, 30, 40, 60, 80, 100, 120, 140], colors='black'  # levels=np.linspace(0, .1, num=6)
)
plt.clabel(CS, inline=True, fmt='%.0f')

tags = g2.predict(vals2)
vals2_0 = vals2[tags == 0]
vals2_1 = vals2[tags == 1]
plt.scatter(vals2_0[:, 0], vals2_0[:, 1], 10, color="red", marker="o")
plt.scatter(vals2_1[:, 0], vals2_1[:, 1], 10, color="blue", marker="^")
plt.show()

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow, linewidth=0, antialiased=True)
ax.set_zlim(0, 100)
ax.zaxis.set_major_locator(LinearLocator(11))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

###########################################################################################################
vals3 = np.array(list(zip(vals[2:], vals[1:], vals)))
g2 = mixture.GaussianMixture(n_components=2, covariance_type="full", tol=1e-3, max_iter=1000, random_state=103,
                             means_init=np.array([[86.29, 86.26, 86.32], [89.08, 89.17, 89.08]]),
                             weights_init=np.array([0.6549, 0.3451]))
g2.fit(vals3)
print(g2.weights_)
print(g2.means_)
print(g2.covariances_)

tags = g2.predict(vals3)
vals3_0 = vals3[tags == 0]
vals3_1 = vals3[tags == 1]
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(vals3_0[:, 0], vals3_0[:, 1], vals3_0[:, 2], alpha=0.8, color="red", marker='o')
ax.scatter3D(vals3_1[:, 0], vals3_1[:, 1], vals3_1[:, 2], alpha=0.8, color="blue", marker='^')
plt.show()
