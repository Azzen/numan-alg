import numpy as np


def mid_point_split(a, b, _):
    return (a + b) / 2


def secant_split(a, b, f):
    return (a * f(b) - b * f(a)) / (f(b) - f(a))


def find_root(f, a, b, epsilon, method, full_output=False):
    """
    Find an approximate solution of the equation f(x) = 0 over ]a,b[, a < b, using simple bisection method

    :param f: the function to find the zero of
    :param a: the lower bound of the initial search interval
    :param b: the upper bound of the initial search interval
    :param epsilon: positive real number of tolerance, e.g. 10**-5
    :param method: the split method we should use: secant_split(a, b, f) or mid_point_split(a, b)
    :param full_output: should we output also the iteration count?
    :return: the mid_point found in the considered interval ]a,b[ and if full_output is true the iteration count.
    """
    iteration_count = 0
    assert a < b, "a should be smaller than b"
    assert f(a) * f(b) < 0, "considered interval is not appropriate for the bisection method!"
    while (b - a) / 2 > epsilon:
        iteration_count += 1
        x_0 = method(a, b, f)
        if iteration_count >= 500:
            print("The exact solution couldn't be approached, the method diverged.")
            return method(a, b, f)
        if f(a) * f(x_0) < 0:
            b = x_0
        elif f(b) * f(x_0) < 0:
            a = x_0
        elif f(x_0) == 0:
            return (x_0, iteration_count) if full_output else x_0
        else:
            print("The exact solution couldn't be approached, the method diverged.")
    return (method(a, b, f), iteration_count) if full_output else method(a, b, f)


def find_roots(f, a, b, step, epsilon, method=mid_point_split):
    """
    Find approximate solutions of the equation f(x) = 0 over ]a,b[, a < b using interval bisection

    :param f: The function to approximate the zeroes of over an interval ]a,b[, a<b
    :param a: the lower bound of the initially considered interval
    :param b: the upper bound of the initially considered interval
    :param step: length of each considered sub intervals
    :param epsilon: The tolerance to apply.
    :param method: the split method we should use: secant_split(f,a,b) or mid_point_split(a,b)
    :return: A list of approximated zeros of the function f
    """
    x = np.arange(a, b + step, step)
    zeros = []
    for i in range(x.shape[0] - 1):
        a_i = x[i]
        b_i = x[i + 1]
        if f(a_i) * f(b_i) < 0:
            zeros.append(find_root(f, a_i, b_i, epsilon, method))
    return zeros


def picard(f, x_0, epsilon, max_iterations):
    # Todo: add documentation
    # Todo: tests
    count = 0
    x_1 = 0
    while count < max_iterations and abs(x_1 - x_0) > epsilon:
        x_1 = f(x_0)
        x_0 = x_1
        count += 1
        return x_0, count


def derivative(f, a, dx=0.005):
    """
    Approximate the derivative of f(a), f'(a) with a given value dx.

    :param f: The function to approximate the derivative f'(a) of
    :param a: a number that'll be evaluated at with f
    :param dx: a number setting the precision of our derivative approximation up.
    :return: the number corresponding to the approximation of f'(a).
    """
    return (f(a+dx) - f(a - dx)) / (2 * dx)


def newton(f, x_0, df, epsilon, max_iterations):
    # Todo: add documentation
    # Todo: tests
    count = 0
    x = x_0
    f_val = float(f(x))
    while abs(f_val) > epsilon and count < max_iterations:
        df_val = df(x)
        assert abs(df_val) > 1e-10, "Division by zero incoming here!"
        x = x - f_val / df_val
        count += 1
        f_val = f(x)
    return x


def parallel(f, x_0, alpha, epsilon, max_iterations):
    # Todo: add documentation
    # Todo: tests
    count = 0
    x = x_0
    f_val = float(f(x))
    while abs(f_val) > epsilon and count < max_iterations:
        x = x - f_val / alpha
        count += 1
        f_val = f(x)
    return x

