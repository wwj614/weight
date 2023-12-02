import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KernelDensity
import scipy.stats as st
from sklearn import mixture
from lmfit import Model
from collections import Counter

vals = np.loadtxt('./c.txt', usecols=(0,), delimiter='\t', unpack=True)
N = vals.shape[0]
x1 = np.linspace(82.9, 92.9, 51, endpoint=True)

########################################################################################################################

kde = KernelDensity(kernel='gaussian', bandwidth=0.3)
kde.fit(vals.reshape(-1, 1))

########################################################################################################################
density = st.gaussian_kde(vals)
density.set_bandwidth(bw_method=density.factor * 0.5)

########################################################################################################################
g2 = mixture.GaussianMixture(n_components=2, covariance_type='spherical', tol=1e-3, max_iter=1000, random_state=103,
                             means_init=np.array([[86.32], [89.08]]),
                             weights_init=np.array([0.599, 0.401]),
                             precisions_init=np.array([1/0.83, 1/1.21]))
g2.fit(vals.reshape(-1, 1))
print("GaussianMixture 2", g2.converged_)
print(g2.weights_[0])
print(g2.means_[0][0])
print(g2.covariances_[0])
print(g2.weights_[1])
print(g2.means_[1][0])
print(g2.covariances_[1])

g3 = mixture.GaussianMixture(n_components=3, covariance_type='spherical', tol=1e-3, max_iter=1000, random_state=103,
                             means_init=np.array([[85], [87], [89]]),
                             weights_init=np.array([0.24, 0.42, 0.34]),
                             precisions_init=np.array([1/0.40, 1/0.59, 1/1.1]))
g3.fit(vals.reshape(-1, 1))
print("GaussianMixture 3", g3.converged_)
print(g3.weights_[0])
print(g3.means_[0][0])
print(g3.covariances_[0])
print(g3.weights_[1])
print(g3.means_[1][0])
print(g3.covariances_[1])
print(g3.weights_[2])
print(g3.means_[2][0])
print(g3.covariances_[2])

print(np.exp(kde.score_samples(x1.reshape(-1, 1))) * N * 0.2)
print(np.exp(g2.score_samples(x1.reshape(-1, 1))) * N * 0.2)
print(np.exp(g3.score_samples(x1.reshape(-1, 1))) * N * 0.2)
print("")

########################################################################################################################
xcnt = Counter({82.0: 0, 93.0: 0})
xcnt.update(vals)
xy = np.array(sorted(xcnt.items()))

'''
xcnt = list({(x, list(vals).count(x)) for x in vals})
xcnt.append((82.0, 0))
xcnt.append((93.0, 0))
xcnt.sort()
xy = np.array(xcnt)
'''

x = xy[:, 0]
y = xy[:, 1].cumsum()
w = np.hstack([np.ones(10) * 10, np.ones(x.size - 20) * 10, np.ones(10) * 10])


def cdf(v, amplitude, center, sigma):
    return amplitude * st.norm.cdf(v, center, sigma)


def pdf(v, amplitude, center, sigma):
    return amplitude * st.norm.pdf(v, center, sigma)


cdfModel1 = Model(cdf, prefix='p1_')
params1 = cdfModel1.make_params(p1_amplitude=300, p1_center=86, p1_sigma=1.3)
params1['p1_amplitude'].min = 50.
result1 = cdfModel1.fit(y, params1, v=x, weights=w)

cdfModel2 = Model(cdf, prefix='p1_') + Model(cdf, prefix='p2_')
params2 = cdfModel2.make_params(p1_amplitude=250, p1_center=86, p1_sigma=1,
                                p2_amplitude=60, p2_center=89, p2_sigma=0.8
                                )
params2['p1_amplitude'].min = 43.
params2['p2_amplitude'].min = 13.
result2 = cdfModel2.fit(y, params2, v=x, weights=w)

cdfModel3 = Model(cdf, prefix='p1_') + Model(cdf, prefix='p2_') + Model(cdf, prefix='p3_')
params3 = cdfModel3.make_params(p1_amplitude=40, p1_center=85, p1_sigma=0.6,
                                p2_amplitude=80, p2_center=87, p2_sigma=0.3,
                                p3_amplitude=180, p3_center=88, p3_sigma=1.4
                                )
params3['p1_amplitude'].min = 13.
params3['p2_amplitude'].min = 13.
params3['p3_amplitude'].min = 13.
result3 = cdfModel3.fit(y, params3, v=x, weights=w)

print(result1.fit_report())
print(result2.fit_report())
print(result3.fit_report())
print(result1.best_fit)
print(result2.best_fit)
print(result3.best_fit)
print(x, '\n', y)

p1_amplitude = result1.best_values["p1_amplitude"]
p1_center = result1.best_values["p1_center"]
p1_sigma = result1.best_values["p1_sigma"]
print([pdf(x, p1_amplitude, p1_center, p1_sigma) * 0.2 for x in x1])

p1_amplitude = result2.best_values["p1_amplitude"]
p1_center = result2.best_values["p1_center"]
p1_sigma = result2.best_values["p1_sigma"]
p2_amplitude = result2.best_values["p2_amplitude"]
p2_center = result2.best_values["p2_center"]
p2_sigma = result2.best_values["p2_sigma"]
print([(pdf(x, p1_amplitude, p1_center, p1_sigma) +
        pdf(x, p2_amplitude, p2_center, p2_sigma)) * 0.2 for x in x1])

p1_amplitude = result3.best_values["p1_amplitude"]
p1_center = result3.best_values["p1_center"]
p1_sigma = result3.best_values["p1_sigma"]
p2_amplitude = result3.best_values["p2_amplitude"]
p2_center = result3.best_values["p2_center"]
p2_sigma = result3.best_values["p2_sigma"]
p3_amplitude = result3.best_values["p3_amplitude"]
p3_center = result3.best_values["p3_center"]
p3_sigma = result3.best_values["p3_sigma"]
print([(pdf(x, p1_amplitude, p1_center, p1_sigma) +
        pdf(x, p2_amplitude, p2_center, p2_sigma) +
        pdf(x, p3_amplitude, p3_center, p3_sigma)) * 0.2 for x in x1])

x2 = np.linspace(82.8, 92.8, 26, endpoint=True)
p1_amplitude = result1.best_values["p1_amplitude"]
p1_center = result1.best_values["p1_center"]
p1_sigma = result1.best_values["p1_sigma"]
print([pdf(x, p1_amplitude, p1_center, p1_sigma) * 0.4 for x in x2])

p1_amplitude = result2.best_values["p1_amplitude"]
p1_center = result2.best_values["p1_center"]
p1_sigma = result2.best_values["p1_sigma"]
p2_amplitude = result2.best_values["p2_amplitude"]
p2_center = result2.best_values["p2_center"]
p2_sigma = result2.best_values["p2_sigma"]
print([(pdf(x, p1_amplitude, p1_center, p1_sigma) +
        pdf(x, p2_amplitude, p2_center, p2_sigma)) * 0.4 for x in x2])

p1_amplitude = result3.best_values["p1_amplitude"]
p1_center = result3.best_values["p1_center"]
p1_sigma = result3.best_values["p1_sigma"]
p2_amplitude = result3.best_values["p2_amplitude"]
p2_center = result3.best_values["p2_center"]
p2_sigma = result3.best_values["p2_sigma"]
p3_amplitude = result3.best_values["p3_amplitude"]
p3_center = result3.best_values["p3_center"]
p3_sigma = result3.best_values["p3_sigma"]
print([(pdf(x, p1_amplitude, p1_center, p1_sigma) +
        pdf(x, p2_amplitude, p2_center, p2_sigma) +
        pdf(x, p3_amplitude, p3_center, p3_sigma)) * 0.4 for x in x2])

plt.scatter(x, y, 2.5, label='data')
plt.plot(x, result1.best_fit, label='best fit1')
plt.plot(x, result2.best_fit, label='best fit2')
plt.plot(x, result3.best_fit, label='best fit3')
plt.legend()
plt.show()
