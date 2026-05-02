import numpy as np
from numba import njit, prange


# =========================
# OPERATORY (Twoje)
# =========================

@njit(fastmath=True, cache=False)
def arc_tan(a, b):
    return a * (2 / np.pi) * np.arctan(b)


@njit(fastmath=True, cache=False)
def sigm(x, exp=5):
    return 1 / (1 + np.exp(-exp * x))


@njit(fastmath=True, cache=False)
def sigmoidal(a, b):
    return a * sigm(b)


@njit(fastmath=True, cache=False)
def ext_sigmoidal(a, b):
    return np.sign(a * b) * sigm(a * b)


@njit(fastmath=True, cache=False)
def log_ext(a, b):
    if a * b == -1:
        return 0
    return np.log(np.abs(a * b + 1))


@njit(fastmath=True, cache=False)
def reduced_log_ext(a, b):
    if a * b == -1:
        return 0
    return max(np.log(np.abs(a * b + 1)), -1)


@njit(fastmath=True, cache=False)
def power_fun(a, exp=0.5):
    return np.sign(a) * np.power(np.abs(a), exp)


# =========================
# OPERATOR SWITCH (numba-safe)
# =========================

@njit(fastmath=True, cache=False)
def apply_operator(op, a, b):
    if op == 0:
        return arc_tan(a, b)
    elif op == 1:
        return ext_sigmoidal(a, b)
    elif op == 2:
        return sigmoidal(a, b)
    elif op == 3:
        return a * power_fun(b)
    elif op == 4:
        return reduced_log_ext(a, b)
    elif op == 5:
        return log_ext(a, b)
    else:
        return a * b


# =========================
# UNIVARIATE
# =========================

@njit(fastmath=True, cache=False)
def _apply_kernel_univariate(X, weights, length, bias, dilation, padding, op):
    n_timepoints = len(X)

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    ppv = 0
    max_val = -np.inf

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        s = bias
        idx = i

        for j in range(length):
            if 0 <= idx < n_timepoints:
                s += apply_operator(op, weights[j], X[idx])
            idx += dilation

        if s > max_val:
            max_val = s
        if s > 0:
            ppv += 1

    return np.float32(ppv / output_length), np.float32(max_val)


# =========================
# MULTIVARIATE (KLUCZOWE)
# =========================

@njit(fastmath=True, cache=False)
def _apply_kernel_multivariate(
    X, weights, length, bias, dilation, padding,
    num_channel_indices, channel_indices
):
    n_channels, n_timepoints = X.shape

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    ppv = 0
    max_val = -np.inf

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        s = bias
        idx = i

        for j in range(length):
            if 0 <= idx < n_timepoints:
                for k in range(num_channel_indices):
                    s += weights[k, j] * X[channel_indices[k], idx]
            idx += dilation

        if s > max_val:
            max_val = s
        if s > 0:
            ppv += 1

    return np.float32(ppv / output_length), np.float32(max_val)


# =========================
# GŁÓWNA TRANSFORMACJA
# =========================

@njit(parallel=True, fastmath=True, cache=False)
def transform(
    X,
    weights,
    lengths,
    biases,
    dilations,
    paddings,
    num_channel_indices,
    channel_indices,
    op
):
    n_instances, n_channels, _ = X.shape
    num_kernels = len(lengths)

    features = np.zeros((n_instances, num_kernels * 2), dtype=np.float32)

    for i in prange(n_instances):
        a1 = 0  # weights index
        a2 = 0  # channel index
        a3 = 0  # feature index

        for j in range(num_kernels):
            b1 = a1 + num_channel_indices[j] * lengths[j]
            b2 = a2 + num_channel_indices[j]

            if num_channel_indices[j] == 1:
                features[i, a3:a3 + 2] = _apply_kernel_univariate(
                    X[i, channel_indices[a2]],
                    weights[a1:b1],
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    op
                )
            else:
                w = weights[a1:b1].reshape(
                    (num_channel_indices[j], lengths[j])
                )

                features[i, a3:a3 + 2] = _apply_kernel_multivariate(
                    X[i],
                    w,
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    num_channel_indices[j],
                    channel_indices[a2:b2]
                )

            a1 = b1
            a2 = b2
            a3 += 2

    return features