import matplotlib.pyplot as plt
import statistics as st
import random
import numpy as np
import scipy.stats as stats

def calc_normal_prob_density(x, dev, M):
    y = -0.5*(1/dev*(x - M))**2
    y = np.exp(y)
    y = (1 / (np.sqrt(2 * np.pi)*dev))*y
    return y

n_count = 1000
random_list = []
for i in range(n_count):
    random_list.append(random.randrange(100))


plt.figure()
n, bins, patches = plt.hist(random_list, bins=20, edgecolor = 'k', density=1) # bins количество столбцов, patches параметры столбцов
mean = st.mean(random_list)
std_dev = st.stdev(random_list)
norm_density = calc_normal_prob_density(bins, std_dev, mean)
plt.plot(bins, norm_density, '--')


plt.figure()
norm_list = mean + std_dev*np.random.randn(n_count) # второе слагаемое это шумовое распределение (шум), которое мы прибавляем к первому, тем самым сгенирировали псевдонормальное распределение
n, bins, patches = plt.hist(norm_list, bins=20, edgecolor = 'k', density=1)
norm_density = calc_normal_prob_density(bins, std_dev, mean)
mean = st.mean(norm_list)
std_dev = st.stdev(norm_list)
plt.plot(bins, norm_density, '--')

plt.show()


def interpret_anderson(anderson_result):
    main_stats = anderson_result.statistic
    for i in range(len(anderson_result.critical_values)):
        critical_value = anderson_result.critical_values[i]
        significance_level = anderson_result.significance_level[i]
        result = 'Accepted'
        if main_stats > critical_value:
            result = 'Declined'
        print('Для уровня значимости {} гипотеза нормальности {}'.format(significance_level, result))


print('Anderson TEST')
anderson_rand = stats.anderson(random_list, dist= 'norm')
print('Для случайной выборки')
interpret_anderson(anderson_rand)
anderson_norm = stats.anderson(norm_list, dist= 'norm')
print('Для нормальной')
interpret_anderson(anderson_norm)

def interpret_shapiro(shapiro_result, level= 0.05):
    if (shapiro_result[1] < level):
        return 'Declined'
    else:
        return 'Accepted'

print('Shapiro TEST')
random_shapiro = stats.shapiro(random_list)
print('Случайная выборка:')
result = interpret_shapiro(random_shapiro)
print(result)
norm_shapiro = stats.shapiro(norm_list)
print('Нормальная выборка:')
result = interpret_shapiro(norm_shapiro)
print(result)