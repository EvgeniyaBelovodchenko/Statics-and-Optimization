import numpy as np
import matplotlib.pyplot as plt
import random
import statistics as st
import scipy.optimize as opt
import scipy.stats as stats


def f(x, a, b, c, d):
    result = 1 + np.exp(-c * (x - d))
    result = a / result
    return result + b

def calc_normal_prob_density(x, dev, M):
    y = -0.5 * (1 / dev * (x - M)) ** 2
    y = np.exp(y)
    y = (1 / (np.sqrt(2 * np.pi) * dev)) * y
    return y


val_a, val_c = np.random.exponential(size=2)
val_b, val_d = np.random.randn(2)
print('Коэффициенты сигмоиды:')
print(val_a, val_b, val_c, val_d, sep=',')

n = 2000

x_values = np.linspace(-20, 20, n)
y_values = f(x_values, val_a, val_b, val_c, val_d)

first_ploy = plt.figure()
#plt.plot(x_values, y_values, '--k', lw = 3, label = 'ideal)
y_noise = y_values + val_a * 0.2 * np.random.randn(n)
plt.plot(x_values, y_noise, 'og', markeredgecolor='k')

(_a, _b, _c, _d), _ = opt.curve_fit(f, x_values, y_noise, (1, 1, 1, 1))
y_fit = f(x_values, _a, _b, _c, _d)

plt.plot(x_values, y_fit, '--', lw=3, label='fit')


plt.grid(True)
plt.legend()

dif = y_fit - y_noise
mean = np.mean(dif)
std = np.std(dif)
n_bins = 50

second_plot = plt.figure(2)
n, bins, patches = plt.hist(dif, n_bins, edgecolor='k', density=True)
norm_dist = calc_normal_prob_density(bins, std, mean)
plt.plot(bins, norm_dist, '--y', label='Плотность вероятности')
plt.legend()
plt.show()


