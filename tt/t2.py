import numpy as np
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt
import GPy

x, y = np.loadtxt('./b.txt', usecols=(0, 2), delimiter='\t', unpack=True)

####################################################################################################################
kernel = GPy.kern.RBF(input_dim=1, variance=1)
model = GPy.models.GPRegression(x.reshape(-1, 1), y.reshape(-1, 1), kernel=kernel)
model.optimize()
ym, _ = model.predict(x.reshape(-1, 1))
print(ym.ravel())
plt.plot(x, ym)
plt.show()


####################################################################################################################
w = np.hstack([np.ones(5) * 10, np.ones(x.size - 10) * 10, np.ones(5) * 10])

# 1 Gaussian:
model1 = GaussianModel(prefix='p1_')
params1 = model1.make_params(p1_amplitude=25, p1_center=87, p1_sigma=1.5)
params1['p1_amplitude'].min = 3.
result1 = model1.fit(y, params1, x=x, weights=w)

# 2 Gaussian:
model2 = GaussianModel(prefix='p1_') + GaussianModel(prefix='p2_')
params2 = model2.make_params(p1_amplitude=20, p1_center=86, p1_sigma=1,
                             p2_amplitude=5, p2_center=89, p2_sigma=0.5)
params2['p1_amplitude'].min = 3.
params2['p2_amplitude'].min = 3.
result2 = model2.fit(y, params2, x=x, weights=w)

# 3 Gaussian:
model3 = GaussianModel(prefix='p1_') + GaussianModel(prefix='p2_') + GaussianModel(prefix='p3_')
params3 = model3.make_params(p1_amplitude=7, p1_center=86, p1_sigma=0.6,
                             p2_amplitude=4, p2_center=87, p2_sigma=0.4,
                             p3_amplitude=13, p3_center=89, p3_sigma=1.3)
params3['p1_amplitude'].min = 3.
params3['p2_amplitude'].min = 1.
params3['p3_amplitude'].min = 3.
result3 = model3.fit(y, params3, x=x, weights=w)

# print out param values, uncertainties, and fit statistics, or get best-fit
# parameters from `result.params`
print(result1.fit_report())
print(result2.fit_report())
print(result3.fit_report())
print(result1.best_fit)
print(result2.best_fit)
print(result3.best_fit)

# plot results
plt.scatter(x, y, label='data')
plt.plot(x, result1.best_fit, label='best fit1')
plt.plot(x, result2.best_fit, label='best fit2')
plt.plot(x, result3.best_fit, label='best fit3')
plt.legend()
plt.show()
