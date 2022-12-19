import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter

def vandermode(x_points, y_points):
    matrix = np.zeros((x_points.size, y_points.size))
    for col in range(0, x_points.size):
        matrix[:, col] = x_points ** col
    matrix = np.flip(matrix, axis=1)
    a = np.linalg.solve(matrix, y_points)
    return a

def f(x, val_a, val_b, val_c):
    result = np.abs(x)**np.abs(val_b)
    result = val_a * result
    return result - val_c

experimental_points = 15
a_, b_, c_ = 2, 3, 4
x_experimental = np.linspace(-1, 4, experimental_points)
x_test = np.linspace(-1, 4, experimental_points*3)

y_ideal = f(x_experimental, a_, b_, c_)
y_experimental = y_ideal + a_ * 2 * np.random.randn(experimental_points)
plt.plot(x_experimental, y_experimental, 'o',
         markerfacecolor='w', markersize=8, label="Exp")


coeff = vandermode(x_experimental, y_experimental)
y_poly = np.polyval(coeff, x_test)
plt.plot(x_test, y_poly, '-k', lw=2, label='Poly')

spline = inter.InterpolatedUnivariateSpline(x_experimental, y_experimental)
y_spline = spline(x_test)
plt.plot(x_test, y_spline, '--r', lw=2, label='Spline')

plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in')


plt.legend()

plt.show()


'''
plt.plot(x_experimental, y_ideal, '-b', label='ideal', lw=2)
#plt.plot(x_experimental, y_experimental, 'o', markerfacecolor='w', markersize=8, label='exp')



x = np.array([2., 3., 4., 5., 6.])
y = np.array([2., 3., 4., 5., 6])
z = np.linspace(2, 6, 20)
polinomial_coeff = vandermode(x, y)
polynom = np.polyval(polinomial_coeff, z)
#print(polinomial_coeff)

plt.plot(x, y, 'o')
plt.plot(z, polynom, '-', label='polynom')

plt.legend()
plt.show()
'''