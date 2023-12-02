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
plt.xticks(np.linspace(83, 93, 11, endpoint=True))
plt.yticks(np.linspace(83, 93, 11, endpoint=True))

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
ax.set_zlim(0, 120)
ax.zaxis.set_major_locator(LinearLocator(13))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_xlim(83, 93)
ax.xaxis.set_major_locator(LinearLocator(6))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_ylim(83, 93)
ax.yaxis.set_major_locator(LinearLocator(6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

#####################################################################################################################

'''
g3 = mixture.GaussianMixture(n_components=3, covariance_type="spherical", tol=1e-3, max_iter=1000, random_state=103,
                             means_init=np.array([[85.37, 85.29], [86.77, 86.81], [88.99, 88.95]]),
                             weights_init=np.array([0.180, 0.561, 0.259]))
g3.fit(vals2)
print(g3.weights_)
print(g3.means_)
print(g3.covariances_)

# Plot the contour.
X, Y = np.meshgrid(x, y)
XY = np.array([X.ravel(), Y.ravel()]).T
Z = np.exp(g3.score_samples(XY)) * 1000
Z = Z.reshape(X.shape)

CS = plt.contour(
    X, Y, Z, levels=[1, 2, 5, 10, 20, 30, 40, 60, 80, 100, 120, 140], colors='black'  # levels=np.linspace(0, .1, num=6)
)
plt.clabel(CS, inline=True, fmt='%.0f')
plt.xticks(np.linspace(83, 93, 11, endpoint=True))
plt.yticks(np.linspace(83, 93, 11, endpoint=True))

tags = g3.predict(vals2)
vals2_0 = vals2[tags == 0]
vals2_1 = vals2[tags == 1]
vals2_2 = vals2[tags == 2]
plt.scatter(vals2_0[:, 0], vals2_0[:, 1], 10, color="red", marker="o")
plt.scatter(vals2_1[:, 0], vals2_1[:, 1], 10, color="blue", marker="^")
plt.scatter(vals2_2[:, 0], vals2_2[:, 1], 10, color="green", marker="*")
plt.show()

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow, linewidth=0, antialiased=True)
ax.set_zlim(0, 120)
ax.zaxis.set_major_locator(LinearLocator(13))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_xlim(83, 93)
ax.xaxis.set_major_locator(LinearLocator(11))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_ylim(83, 93)
ax.yaxis.set_major_locator(LinearLocator(11))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
'''
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
ax.set_zlim(84, 92)
ax.zaxis.set_major_locator(LinearLocator(11))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_xlim(84, 92)
ax.xaxis.set_major_locator(LinearLocator(11))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_ylim(84, 92)
ax.yaxis.set_major_locator(LinearLocator(11))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.show()
