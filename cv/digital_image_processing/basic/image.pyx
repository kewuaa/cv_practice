# -*- coding: utf-8 -*-
#******************************************************************#
#
#                     Filename: image.pyx
#
#                       Author: kewuaa
#                      Created: 2022-05-22 19:17:17
#                last modified: 2022-05-26 18:51:25
#******************************************************************#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from libc.math cimport pow as c_pow
from libc.math cimport exp as c_exp
from libc.math cimport log2 as c_log2
from libc.math cimport pi
from cpython.mem cimport PyMem_Malloc
from cpython.mem cimport PyMem_Free
from cython.parallel cimport prange
cimport numpy as cnp
import numpy as np


cdef extern from '<complex.h>' nogil:
    double complex cexp(double complex z)
    double complex cpow(double complex x, double complex y)


cdef fused double_or_double_complex_arr1d:
    double[:]
    double complex[:]


cdef void *dft(double complex[:] result_mem, double_or_double_complex_arr1d array):
    """离散傅里叶变换.

    Args:
    result_mem:
        存放结果的数组视图(需初始化为零)
    array:
        需要进行变换的数组

    Returns:
        None
    """

    cdef:
        unsigned int N = array.shape[0]
        unsigned int i, j, k
        double complex J = -2j * pi / N
        double complex w
    for k in prange(N, nogil=True):
        w = J * k
        for i in range(N):
            result_mem[k] += array[i] * cexp(i * w)


cdef void *idft(double complex[:] result_mem, double_or_double_complex_arr1d array):
    """离散傅里叶变换逆过程.

    Args:
    result_mem:
        存放结果的数组视图(需初始化为零)
    array:
        需要进行逆变换的数组

    Returns:
        None
    """

    cdef:
        unsigned int N = array.shape[0]
        unsigned int i, j, k
        double complex J = 2j * pi / N
        double complex w
    for k in prange(N, nogil=True):
        w = J * k
        for i in range(N):
            result_mem[k] += array[i] * cexp(i * w)
        result_mem[k] /= N


cdef double complex[:] fft_(double complex[:] result_mem, double_or_double_complex_arr1d array) nogil:
    """快速傅里叶变换.

    Args:
    result_mem:
        存放结果的数组视图
    array:
        需要进行变换的数组

    Returns:
        结果数组(其实和输入的用于存放结果的数组为同一数组)
    """

    cdef:
        unsigned int i
        unsigned int half
        unsigned int N = array.shape[0]
        double complex w, x, e, o
        double complex[:] array_e, array_o
    # 无法拆成奇偶对形式, 停止递归, 返回结果
    if N <= 1:
        result_mem[0] = <double complex> array[0]
        return result_mem
    half = N / 2
    w = -2j * pi / N
    array_e = fft_(result_mem[: half], array[::2])
    array_o = fft_(result_mem[half:], array[1::2])
    # x1 + x2 = 0, x1² = x2²
    # P(x1) = Pe(x1²) + po(x1²)
    # p(x2) = pe(x2²) - po(x2²)
    # 根据公式依次带回
    for i in range(half):
        x = cexp(w * i)
        e = array_e[i]
        o = array_o[i]
        result_mem[i] = e + o * x
        result_mem[i + half] = e - o * x
    return result_mem


cdef double complex[:] ifft_(double complex[:] result_mem, double_or_double_complex_arr1d array) nogil:
    """快速傅里叶逆变换.

    Args:
    result_mem:
        存放结果的数组视图
    array:
        需要进行逆变换的数组

    Returns:
        结果数组(其实和输入的用于存放结果的数组为同一数组)
    """

    cdef:
        unsigned int i
        unsigned int half
        unsigned int N = array.shape[0]
        double complex w, x, e, o
        double complex[:] array_e, array_o
    if N <= 1:
        result_mem[0] = <double complex> array[0]
        return result_mem
    half = N / 2
    w = 2j * pi / N
    array_e = ifft_(result_mem[: half], array[::2])
    array_o = ifft_(result_mem[half:], array[1::2])
    for i in range(half):
        x = cexp(w * i)
        e = array_e[i] / 2
        o = array_o[i] / 2
        result_mem[i] = e + o * x
        result_mem[i + half] = e - o * x
    return result_mem


cpdef cnp.ndarray[double complex, ndim=1] fft(double[:] array):
    """一维离散傅里叶变换.

    Args:
    array:
        需要进行变换的数组

    Returns:
        返回变换结果
    """

    cdef:
        int N = array.shape[0]
        double complex[::1] mem
        cnp.ndarray[double complex, ndim=1] result = np.zeros(N, dtype=complex)
    mem = result
    if c_log2(<double> N) % 1 == 0:
        with nogil:
            fft_[double[:]](mem, array)
    else:
        dft[double[:]](mem, array)
    return result


cpdef cnp.ndarray[double complex, ndim=1] ifft(double complex[:] array):
    """一维离散傅里叶逆变换.

    Args:
    array:
        需要进行逆变换的数组

    Returns:
        返回逆变换结果
    """

    cdef:
        int N = array.shape[0]
        double complex[::1] mem
        cnp.ndarray[double complex, ndim=1] result = np.zeros(N, dtype=complex)
    mem = result
    if c_log2(<double> N) % 1 == 0:
        with nogil:
            ifft_[complex[:]](mem, array)
    else:
        idft[complex[:]](mem, array)
    return result


cpdef cnp.ndarray[double complex, ndim=2] fft2(double[:, ::1] array):
    """二维离散傅里叶变换.

    Args:
    array:
        需要进行变换的数组

    Returns:
        返回变换结果
    """

    cdef:
        unsigned int i
        unsigned int rows = array.shape[0]
        unsigned int cols = array.shape[1]
        double complex[:, ::1] mem
        double complex *temp = <double complex *>PyMem_Malloc(rows * cols * sizeof(double complex))
        double complex[:, ::1] temp_mem = <double complex[: rows, : cols]> temp
        cnp.ndarray[double complex, ndim=2] result = np.zeros([rows, cols], dtype=complex)
    mem = result
    temp_mem[:] = 0 + 0j
    if c_log2(cols) % 1 == 0:
        for i in prange(rows, nogil=True):
            fft_[double[:]](temp_mem[i, ...], array[i, ...])
    else:
        for i in range(rows):
            dft[double[:]](temp_mem[i, ...], array[i, ...])

    if c_log2(rows) % 1 == 0:
        for i in prange(cols, nogil=True):
            fft_[complex[:]](mem[..., i], temp_mem[..., i])
    else:
        for i in range(cols):
            dft[complex[:]](mem[..., i], temp_mem[..., i])
    PyMem_Free(temp)
    return result


cpdef cnp.ndarray[double complex, ndim=2] ifft2(double complex[:, ::1] array):
    """二维离散傅里叶逆变换.

    Args:
    array:
        需要进行逆变换的数组

    Returns:
        返回逆变换结果
    """

    cdef:
        unsigned int i
        unsigned int rows = array.shape[0]
        unsigned int cols = array.shape[1]
        double complex[:, ::1] mem
        double complex *temp = <double complex *>PyMem_Malloc(rows * cols * sizeof(double complex))
        double complex[:, ::1] temp_mem = <double complex[: rows, : cols]> temp
        cnp.ndarray[double complex, ndim=2] result = np.zeros([rows, cols], dtype=complex)
    mem = result
    temp_mem[:] = 0 + 0j
    if c_log2(cols) % 1 == 0:
        for i in prange(rows, nogil=True):
            ifft_[complex[:]](temp_mem[i, ...], array[i, ...])
    else:
        for i in range(rows):
            idft[complex[:]](temp_mem[i, ...], array[i, ...])

    if c_log2(rows) % 1 == 0:
        for i in prange(cols, nogil=True):
            ifft_[complex[:]](mem[..., i], temp_mem[..., i])
    else:
        for i in range(cols):
            idft[complex[:]](mem[..., i], temp_mem[..., i])
    PyMem_Free(temp)
    return result


cdef inline double mul_sum(double[:, :] arr1, double[:, ::1] arr2) nogil:
    """数组对应项乘积最后求和.

    Args:
    arr1:
        输入数组1
    arr2:
        输入数组2

    Returns:
        返回结果
    """

    cdef:
        unsigned int i, j
        unsigned int rows = arr1.shape[0]
        unsigned int cols = arr1.shape[1]
        double s = 0.
    for i in range(rows):
        for j in range(cols):
            s += arr1[i, j] * arr2[i, j]
    return s


cdef void *init_gaussian_kernel(double[:, ::1] mem, double sigma) nogil:
    """初始化高斯核, 对其进行赋值.

    Args:
    mem:
        高斯核内存视图
    sigma:
        标准差

    Returns:
        None
    """

    cdef:
        unsigned int kernel_row = mem.shape[0]
        unsigned int kernel_col = mem.shape[1]
        unsigned int row_move_distance = kernel_row / 2
        unsigned int col_move_distance = kernel_col / 2
        unsigned int i, j
        double s = 0.
        double modulus
    for i in range(kernel_row):
        for j in range(kernel_col):
            modulus = c_pow(<double> i - row_move_distance, 2.) + c_pow(<double> j - col_move_distance, 2.)
            mem[i, j] = c_exp(-modulus / c_pow(sigma, 2.) / 2)
            s += mem[i, j]
    for i in range(kernel_row):
        for j in range(kernel_col):
            mem[i, j] /= s


cdef fused Image:
    double[:, ::1]
    double[:, :, ::1]


cpdef cnp.ndarray spacial_filter(Image img, double[:, ::1] kernel):
    """空域滤波器.

    Args:
    img:
        输入图像
    kernel:
        滤波核

    Returns:
        返回滤波结果
    """

    cdef:
        unsigned int kernel_row = kernel.shape[0]
        unsigned int kernel_col = kernel.shape[1]
        unsigned int rows = img.shape[0] - kernel_row + 1
        unsigned int cols = img.shape[1] - kernel_col + 1
        unsigned int channels = img.shape[2]
        unsigned int i, j, k
        double[:, :, ::1] mem
        cnp.ndarray result
    if Image is double[:, ::1]:
        channels = 1
    result = np.empty([rows, cols, channels])
    mem = result

    if Image is double[:, ::1]:
        with nogil:
            for i in range(rows):
                for j in range(cols):
                    mem[i, j, 0] = mul_sum(img[i: i + kernel_row,
                                               j: j + kernel_col], kernel)
        return result[..., 0]
    else:
        for k in prange(channels, nogil=True):
            for i in range(rows):
                for j in range(cols):
                    mem[i, j, k] = mul_sum(img[..., k][i: i + kernel_row,
                                                       j: j + kernel_col], kernel)
        return result


cpdef cnp.ndarray gaussian_filter(Image img, int kernel_size, double sigma):
    """高斯滤波.

    Args:
    img:
        输入图像
    kernel_size:
        滤波核大小
    sigma:
        标准差

    Returns:
        返回滤波结果
    """

    cdef:
        double *kernel = <double *> PyMem_Malloc(kernel_size * kernel_size * sizeof(double))
        double[:, ::1] kernel_mem = <double[: kernel_size, : kernel_size]> kernel
        cnp.ndarray result
    init_gaussian_kernel(kernel_mem, sigma)
    result = spacial_filter(img, kernel_mem)
    PyMem_Free(kernel)
    return result


cpdef cnp.ndarray mean_filter(Image img, int kernel_size):
    """均值滤波.

    Args:
    img:
        输入图像
    kernel_size:
        滤波核大小

    Returns:
        返回滤波结果
    """

    cdef:
        int size = kernel_size * kernel_size
        double *kernel = <double *> PyMem_Malloc(size * sizeof(double))
        double[:, ::1] kernel_mem = <double[: kernel_size, : kernel_size]> kernel
        cnp.ndarray result
    kernel_mem[:] = 1. / <double> size
    result = spacial_filter(img, kernel_mem)
    PyMem_Free(kernel)
    return result


cdef void *fast_median_filter(double[:, :] result_mem, double[:, :] img, unsigned int kernel_size) nogil:
    """快速中值滤波.

    Args:
    result_mem:
        存放结果的数组视图
    img:
        输入图像(单通道)
    kernel_size:
        滤波核大小

    Returns:
        None
    """

    cdef:
        int cum_sum = 0
        int left, right
        int threshold = kernel_size * kernel_size / 2
        int histongram[256]
        int[::1] histongram_mem
        unsigned int rows = result_mem.shape[0]
        unsigned int cols = result_mem.shape[1]
        unsigned int i, j, k, l
        double [:, :] window_mem
        double median = 0.
    with gil:
        histongram_mem = histongram
    for i in range(rows):
        for j in range(cols):
            if j == 0:
                window_mem = img[i: kernel_size, j: kernel_size]
                cum_sum = 0
                # 初始化直方图
                histongram_mem[:] = 0
                # 更新直方图
                for k in range(kernel_size):
                    for l in range(kernel_size):
                        histongram_mem[<int> window_mem[k, l]] += 1
                # 通过计算累计直方图得到中值
                for k in range(256):
                    cum_sum += histongram_mem[k]
                    if cum_sum >= threshold:
                        median = <double> k + 1
                        break
            else:
                # 减去最左边的一列, 加上最右边一列
                for k in range(kernel_size):
                    left = <int> img[i + k, j - 1]
                    histongram_mem[left] -= 1
                    if left < median:
                        cum_sum -= 1
                    right = <int> img[i + k, j + kernel_size - 1]
                    histongram_mem[right] += 1
                    if right < median:
                        cum_sum += 1
                # 小于中值个数小于阈值则以当前中值为起点继续累加直到cum_sum超过阈值
                if cum_sum < threshold:
                    for k in range(<unsigned int> median, 256):
                        cum_sum += histongram_mem[k]
                        if cum_sum >= threshold:
                            median = <double> k + 1
                            break
                # 小于中值个数大于阈值则以当前中值减一为起点累减知道cum_sum回到阈值
                elif cum_sum > threshold:
                    for k in range(<unsigned int> median - 1, -1, -1):
                        cum_sum -= histongram_mem[k]
                        if cum_sum <= threshold:
                            median = <double> k
                            break
            result_mem[i, j] = median


cpdef cnp.ndarray median_filter(Image img, int kernel_size):
    """中值滤波.

    Args:
    img:
        输入图像
    kernel_size:
        滤波核大小

    Returns:
        返回滤波结果
    """

    cdef:
        unsigned int i
        unsigned int rows = img.shape[0] - kernel_size + 1
        unsigned int cols = img.shape[1] - kernel_size + 1
        unsigned int channels = img.shape[2]
        double[:, :, ::1] result_mem
        cnp.ndarray result
    if Image is double[:, ::1]:
        channels = 1
    result = np.empty([rows, cols, channels])
    result_mem = result
    if Image is double[:, ::1]:
        with nogil:
            fast_median_filter(result_mem[..., 0], img, kernel_size)
        result = np.array(result_mem[..., 0])
    else:
        for i in prange(channels, nogil=True):
            fast_median_filter(result_mem[..., i], img[..., i], kernel_size)
        result = np.array(result_mem)
    return result

