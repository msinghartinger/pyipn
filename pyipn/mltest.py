# import numpy as np


# class NWE(object):
#     def __init__(self, y_a, sigma_a, times_a, y_b, sigma_b, times_b, h):
#         self.a = {
#         'y': y_a,
#         'sigma': sigma_a,
#         'times': times_a
#         }
#         self.b = {
#         'y': y_b,
#         'sigma': sigma_b,
#         'times': times_b
#         }
#         self.h = h

#     def gaussian_kernel(self, t, k, times):
#         return np.exp(-(t-times[k])**2/(times[k+self.h]-times[k-self.h])**2)

#     def f_A(self, t):
#         num = 0.
#         den = 0.
#         for k, y_ak in enumerate(self.a['y']):
#             num += y_ak * self.gaussian_kernel(t, k, self.a['times'])
#             den += self.gaussian_kernel(t, k, self.a['times'])
#         return num/den

#     def f_B(self, t, delta, M):
#         return M*self.f_A(t-delta)

#     def er(self, h, delta):
#         for k in 

