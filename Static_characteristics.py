import matplotlib.pyplot as plt
import  statistics as st
import random
import numpy as np
import scipy.stats as stats



def calc_mean_value(numb):
    total = sum(numb)
    return total/len(numb)

def calc_median(number):
    middle = int(len(number) / 2)
    sorted_number = sorted(number)
    if len(number) % 2 == 0:
        return (sorted_number[middle]+sorted_number[middle - 1]) / 2
    else:
        return sorted_number[middle]

def calc_stdev(numb):
    diffs = 0
    avg = calc_mean_value(numb)
    for i in numb:
        diffs += (i - avg) ** 2
    result = (diffs/ (len(numb)-  1)) ** 0.5
    return result

number_count = 500
max_value = 500
values = []

for i in range(number_count):
    value = random. randrange(max_value)
    values.append(value)


mean_value = calc_mean_value(values)
mean_python = st.mean(values)
print('Mean value = {:.2f}'. format(mean_value))
print('Second Mean value = {:.2f}'. format(mean_python))

median_value = calc_median(values)
print('Median = ', median_value)
std_value = calc_stdev(values)
print('Standart deviation =', round(std_value, 2))

plt.plot(values, 'or', markersize=5, markeredgecolor = 'k')
plt.axhline(y = mean_value, color = 'k', linestyle='-', label ='Mean')
plt.axhline(y = median_value, color = 'b', linestyle=':', label ='Median')
plt.axhline(y = mean_value - std_value, color = 'g', linestyle='-', label ='Std')
plt.axhline(y = mean_value + std_value, color = 'g', linestyle='-')
plt.title('Scatter')
plt.xlabel('x_values')
plt.ylabel('numbers')
plt.grid(True, linestyle = ':')
plt.legend()
plt.show()

