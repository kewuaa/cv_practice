# -*- coding: utf-8 -*-
#******************************************************************#
#
#                     Filename: image.pyx
#
#                       Author: kewuaa
#                      Created: 2022-05-22 19:17:17
#                last modified: 2022-05-27 23:36:13
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


cdef class DoubleMemory:
    cdef:
        unsigned int _size
        double *_data
        double[:, ::1] view2d

    def __cinit__(self, unsigned int size):
        self._data = <double *> PyMem_Malloc(size * sizeof(double))
        if self._data is NULL:
            raise MemoryError()

    def __init__(self, unsigned int size):
        self._size = size

    def __dealloc__(self):
        PyMem_Free(self._data)

    cdef double[:, ::1] init_2dview(self, unsigned int rows, unsigned int cols):
        self.view2d = <double[: rows, : cols]> self._data
        return self.view2d


cdef class DoubleComplexMemory:
    cdef:
        unsigned int _size
        double complex *_data
        double complex[:, ::1] view2d

    def __cinit__(self, unsigned int size):
        self._data = <double complex *> PyMem_Malloc(size * sizeof(double complex))
        if self._data is NULL:
            raise MemoryError()

    def __init__(self, unsigned int size):
        self._size = size

    def __dealloc__(self):
        PyMem_Free(self._data)

    cdef double complex[:, ::1] init_2dview(self, unsigned int rows, unsigned int cols):
        self.view2d = <double complex[: rows, : cols]> self._data
        return self.view2d


cdef fused double_or_double_complex_arr1d:
    double[:]
    double complex[:]


cdef void *dft(double complex[:] result_mem, double_or_double_complex_arr1d array) nogil:
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
    for k in range(N):
        w = J * k
        for i in range(N):
            result_mem[k] += array[i] * cexp(i * w)


cdef void *idft(double complex[:] result_mem, double_or_double_complex_arr1d array) nogil:
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
    for k in range(N):
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
        ifft_[complex[:]](mem, array)
    else:
        idft[complex[:]](mem, array)
    return result


cdef DoubleComplexMemory fft2_(double[:, :] array):
    """二维离散傅里叶变换.

    Args:
    array:
        需要进行变换的数组

    Returns:
        返回变换结果
    """

    cdef:
        unsigned int i, j
        unsigned int rows = array.shape[0]
        unsigned int cols = array.shape[1]
        unsigned int size = rows * cols
        DoubleComplexMemory temp = DoubleComplexMemory(size)
        DoubleComplexMemory result = DoubleComplexMemory(size)
        double complex[:, ::1] temp_mem = temp.init_2dview(rows, cols)
        double complex[:, ::1] mem = result.init_2dview(rows, cols)
    temp_mem[:] = 0 + 0j
    if c_log2(cols) % 1 == 0:
        for i in prange(rows, nogil=True):
            fft_[double[:]](temp_mem[i, ...], array[i, ...])
    else:
        for i in prange(rows, nogil=True):
            dft[double[:]](temp_mem[i, ...], array[i, ...])

    if c_log2(rows) % 1 == 0:
        for i in prange(cols, nogil=True):
            fft_[complex[:]](mem[..., i], temp_mem[..., i])
    else:
        for i in prange(cols, nogil=True):
            dft[complex[:]](mem[..., i], temp_mem[..., i])
    return result


cpdef cnp.ndarray[double complex, ndim=2] fft2(double[:, :] array):
    cdef DoubleComplexMemory result = fft2_(array)
    return np.array(result.view2d, dtype=complex)


cdef DoubleComplexMemory ifft2_(double complex[:, :] array):
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
        unsigned int size = rows * cols
        DoubleComplexMemory temp = DoubleComplexMemory(size)
        DoubleComplexMemory result = DoubleComplexMemory(size)
        double complex[:, ::1] mem = result.init_2dview(rows, cols)
        double complex[:, ::1] temp_mem = temp.init_2dview(rows, cols)
    temp_mem[:] = 0 + 0j
    if c_log2(cols) % 1 == 0:
        for i in prange(rows, nogil=True):
            ifft_[complex[:]](temp_mem[i, ...], array[i, ...])
    else:
        for i in prange(rows, nogil=True):
            idft[complex[:]](temp_mem[i, ...], array[i, ...])

    if c_log2(rows) % 1 == 0:
        for i in prange(cols, nogil=True):
            ifft_[complex[:]](mem[..., i], temp_mem[..., i])
    else:
        for i in prange(cols, nogil=True):
            idft[complex[:]](mem[..., i], temp_mem[..., i])
    return result


cpdef cnp.ndarray[double complex, ndim=2] ifft2(double complex[:, :] array):
    cdef DoubleComplexMemory result = ifft2_(array)
    return np.array(result.view2d, dtype=complex)


cpdef cnp.ndarray[double, ndim=2] spacial_filter(double[:, :] img, double[:, :] kernel):
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
        unsigned int rows = img.shape[0]
        unsigned int cols = img.shape[1]
        unsigned i, j
        int kernel_rows = kernel.shape[0]
        int kernel_cols = kernel.shape[1]
        int size = rows * cols * sizeof(double)
        DoubleMemory kernel_padded = DoubleMemory(size)
        DoubleComplexMemory fft_img, temp
        double complex[:, ::1] temp_view1, temp_view2
    kernel_padded.init_2dview(rows, cols)
    kernel_padded.view2d[:] = 0
    kernel_padded.view2d[: kernel_rows , : kernel_cols] = kernel[::-1, ::-1]
    temp = fft2_(kernel_padded.view2d)
    temp_view1 = temp.view2d
    fft_img = fft2_(img)
    temp_view2 = fft_img.view2d
    for i in range(rows):
        for j in range(cols):
            temp_view2[i, j] *= temp_view1[i, j]
    fft_img = ifft2_(temp_view2)
    return np.abs(fft_img.view2d[kernel_rows / 2:, kernel_cols / 2:])


cdef DoubleMemory init_gaussian_kernel(unsigned int kernel_row, unsigned int kernel_col, double sigma):
    """初始化高斯核, 对其进行赋值.

    Args:
    kernel_row:
        核行数
    kernel_col:
        核列数
    sigma:
        标准差

    Returns:
        None
    """

    cdef:
        DoubleMemory kernel = DoubleMemory(kernel_row * kernel_col)
        double[:, ::1] kernel_mem = kernel.init_2dview(kernel_row, kernel_col)
        unsigned int row_move_distance = kernel_row / 2
        unsigned int col_move_distance = kernel_col / 2
        unsigned int i, j
        double s = 0.
        double modulus
    for i in range(kernel_row):
        for j in range(kernel_col):
            modulus = c_pow(<double> i - row_move_distance, 2.) + c_pow(<double> j - col_move_distance, 2.)
            kernel_mem[i, j] = c_exp(-modulus / c_pow(sigma, 2.) / 2)
            s += kernel_mem[i, j]
    for i in range(kernel_row):
        for j in range(kernel_col):
            kernel_mem[i, j] /= s
    return kernel


cpdef cnp.ndarray[double, ndim=2] gaussian_filter(double[:, :] img, int kernel_size, double sigma):
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
        DoubleMemory kernel
        cnp.ndarray[double, ndim=2] result
    kernel = init_gaussian_kernel(kernel_size, kernel_size, sigma)
    result = spacial_filter(img, kernel.view2d)
    return result


cpdef cnp.ndarray[double, ndim=2] mean_filter(double[:, :] img, int kernel_size):
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
        DoubleMemory kernel = DoubleMemory(size)
        double[:, ::1] kernel_mem = kernel.init_2dview(kernel_size, kernel_size)
        cnp.ndarray[double, ndim=2] result
    kernel_mem[:] = 1. / <double> size
    result = spacial_filter(img, kernel_mem)
    return result


cpdef cnp.ndarray[double, ndim=2] median_filter(double[:, :] img, unsigned int kernel_size):
    """快速中值滤波.

    Args:
    img:
        输入图像(单通道)
    kernel_size:
        滤波核大小

    Returns:
        结果数组
    """

    cdef:
        int cum_sum = 0
        int left, right
        int threshold = kernel_size * kernel_size / 2
        int histongram[256]
        int[::1] histongram_mem
        unsigned int rows = img.shape[0] - kernel_size + 1
        unsigned int cols = img.shape[1] - kernel_size + 1
        unsigned int i, j, k, l
        double[:, ::1] result_mem
        double [:, :] window_mem
        double median = 0.
        cnp.ndarray[double, ndim=2] result = np.empty([rows, cols])
    result_mem = result
    histongram_mem = histongram
    with nogil:
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
                        elif left == median:
                            for l in range(<unsigned int> median, 256):
                                if histongram_mem[l] != 0:
                                    median = <double> l
                                    break
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
                    # 小于中值个数大于阈值则以当前中值减一为起点累减直到cum_sum回到阈值
                    elif cum_sum > threshold:
                        for k in range(<unsigned int> median - 1, -1, -1):
                            cum_sum -= histongram_mem[k]
                            if cum_sum <= threshold:
                                median = <double> k
                                break
                result_mem[i, j] = median
    return result

