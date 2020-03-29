import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def gauss_map(np.ndarray[np.float_t, ndim=2] img,
              np.ndarray[np.float_t, ndim=2] rr,
              np.ndarray[np.float_t, ndim=2] phi,
              np.ndarray[np.float_t, ndim=1] rs,
              np.ndarray[np.float_t, ndim=1] ws,
              np.ndarray[np.float_t, ndim=1] a_s,
              np.ndarray[np.float_t, ndim=1] a_s_std,
              np.ndarray[np.float_t, ndim=1] intensities
              ):
    for i in range(intensities.size):
        img += np.exp(- 8 * (rr - rs[i]) ** 2 / ws[i] ** 2 - 8 * (phi - a_s[i]) ** 2 / a_s_std[i] ** 2) * intensities[i]
    return img
    # return cgauss_map(img, rr, phi,
    #                   rs, ws, a_s, a_s_std,
    #                   intensities)

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef np.ndarray[np.float_t, ndim=2] cgauss_map(
#         np.ndarray[np.float_t, ndim=2] img,
#         np.ndarray[np.float_t, ndim=2] rr,
#         np.ndarray[np.float_t, ndim=2] phi,
#         np.ndarray[np.float_t, ndim=1] rs,
#         np.ndarray[np.float_t, ndim=1] ws,
#         np.ndarray[np.float_t, ndim=1] a_s,
#         np.ndarray[np.float_t, ndim=1] a_s_std,
#         np.ndarray[np.float_t, ndim=1] intensities
# ):
#     for intensity, r, w, a, a_std in zip(intensities, rs, ws, a_s, a_s_std):
#         gauss = np.exp(- 8 * (rr - r) ** 2 / w ** 2 - 8 * (phi - a) ** 2 / a_std ** 2)
#         img += gauss * intensity
#     return img
