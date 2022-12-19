import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.optimize as opt
import scipy.stats as st
import scipy.interpolate as inter


def calc_confidence_interval(data_set, confidence=0.95):
    n = data_set.size
    t_stud = st.t.ppf((1 + confidence) / 2., n - 1)
    mean = np.mean(data_set)
    sem = st.sem(data_set) #нормированная/standa ошибка
    return mean, sem * t_stud

def remove_outliers(data_set):
    z_score = st.zscore(data_set)
    not_outliers = np.where(np.abs(z_score) < 3, True, False)
    cleaned = data_set[not_outliers]
    return cleaned

def f(x, a , b, c, d):
    return a*x**3 + b*x**2 + c*x +d

data_in_set = 20
x_val = np.linspace(-20, 20, data_in_set)
_a, _b, _c, _d = (0.01, 0.02, 0.03, 0.04)
y_val = f(x_val, _a, _b, _c, _d)

first_plot = plt.figure(1)
plt.plot(x_val, y_val, '-r', label='Ideal')

multiplier = np.random.randn(1)
y_noise = y_val + 10 * multiplier * np.random.randn(data_in_set)
plt.plot(x_val, y_noise, '--or', label='Noise')

plt.show()

set_counter = 10
sets = np.zeros((set_counter, data_in_set))

for i in range(set_counter):
    multiplier = np.random.randn(1)
    y_noise = y_val + 10 * multiplier * np.random.randn(data_in_set)
    plt.plot(x_val, y_noise, 'o', label='Data set' + str(i+1))
    sets[i] = y_noise

#print(sets)
#plt.show()

set_1 = sets[:, 0]
#print(set_1)
cleaned_set = remove_outliers(set_1)

y_means = np.zeros(data_in_set)
y_errors = np.zeros(data_in_set)

for set_num in range(sets.shape[1]):
    correct_set = sets[:, set_num]
    cleaned_set = remove_outliers(correct_set)
    x_value = x_val[set_num]
    y_mean, y_error = calc_confidence_interval(cleaned_set)
    #print('For x = {:.2f} y = {:.2f} +/- {:.2f}'.format(x_val, y_mean, y_error))
    y_means[set_num] = y_mean
    y_errors[set_num] = y_error


second_plot = plt.figure(2)
plt.errorbar(x_val, y_means, yerr=y_error, fmt='or', label='mean')


(a_fit, b_fit, c_fit, d_fit), _ = opt.curve_fit(f, x_val, y_val, (1, 1, 1, 1))
y_fit = f(x_val, a_fit, b_fit, c_fit, d_fit)
plt.plot(x_val, y_fit, '--r', label='Fit')
plt.legend()
plt.show()
