"""
Created on 8/7/20

@author: jordanshomefolder
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def deriv(x, func):  # works for analytic functions !
    h = 1e-5
    return (func(x + h) - func(x - h)) / (
                2 * h)  # using the symmetric derivative


def sin(x):
    return np.sin(x)


def d_sin(x):
    return np.cos(x)


def comp_sol(x):
    return x ** 3 - 1


def d_com_sol(x):
    return (3 * x) ** 2


def real_sol(x):
    return np.exp(x) - 5 * x


def d_real_sol(x):
    return np.exp(x) - 5


def newtons_method(func, d_func, x0, Print=False, step_lim=20,
                   tol=1e-10):  ###default parameter to see column info######
    """takes in the function, manually computed deriv, starting point
        step_lim is how many iterations you want the method to take
        tol is how exact you want your error of the sol to be (distance from zero),
            note we use this tol value elsewhere where it is needed

        The function returns the final step value, and arrays of the difference
            steps and other relevant information
        If Print == True the you can see the step and diff to the next step at every iteration
        """

    diff = 1  # arbitraty value
    step = 0
    step_array = []
    diff_array = []
    while diff > tol and step < step_lim:
        step += 1
        if d_func(x0) == 0:
            print(f"function has zero derivative at {x0}")
            return
        x1 = x0 - (func(x0) / d_func(x0))  # solve for next point
        diff = abs(x0 - x1)
        if Print == True:
            print(f"{x0:.3f} , {diff:.3f}")
        step_array.append(x0)
        diff_array.append(diff)
        x0 = x1
    if diff > tol:
        print(
            "failure to find root \n number of steps is \n {} n With the final step of {}".format(
                step, x0))
        return
    else:
        return x0, step

def data_construct(a, b, point, func, d_func, x0, Print, step_lim, tol):
    '''
     #a and b define the box that contains the data, point is the number of points used to graph
    '''
    # I_arr = np.array([[0]*point]*point) - **not sure how to implement I or what it really represents
    xarr = np.array([0]*point)
    steps = np.array([0]*point)
    rootarr = np.array([0]*point)
    yarr = np.array([0]*point)
    fac = b-a/point #equal spaces between bounds
    for i in range(1, point+1):
        f = x0 + i*fac #how do i allow this to be complex? getting an error
        xarr[i+1] = f
        yarr[i-1] = lambda f: f ** 2 +2
        rootarr[i-1] = newtons_method(func, d_func, f, Print, step_lim, tol)[0]
        steps[i-1] = newtons_method(func, d_func, f, Print, step_lim, tol)[1]
    return xarr, yarr, steps, rootarr

def plot():
    # xarr, yarr, steps, rootarr = data_construct(a, b, point, func, d_func, x0, Print, step_lim, tol)
    Z = rootarr #should be I_arr 
    z_min = min([min(Z[i]) for i in range(len(Z))])
    z_max = max([max(Z[i]) for i in range(len(Z))])
    fig, ax0 = plt.subplots(1, 1)
    c = ax0.pcolor(Z, cmap='RdBu', vmin=z_min, vmax=z_max) #<-- 2D array but not sure how to get it (same issue with I)
    ax0.set_title("Newton's Method")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    def f(x):
        return x ** 2 + 2


    res = newtons_method(lambda x: x ** 2 +2, lambda x: 2 * x, 1j + 2, Print=False)
    # print(res)
    a= -1j
    b= 1j
    point = 10
    # print(data_construct(a, b, point, lambda x: x ** 2 +2, lambda x: 2 * x, 1j + 2, Print=False, step_lim=10, tol=1e-10))
    plot()
# diff between next and previous point decreses by ^2 value
# if the root cannot be found at an example where it reaches the min (where the root would be)\
# and slope approaches 0 it changes violently. look at the steps x takes for\
# newtons_method(lambda x: x**2 + 1,lambda x: 2*x, 5)
# Note that if the functon only has real or complex roots you must input a real or complex
# value respectively for the solution to converge to the root


##Can you go to imaginary input to real solution

# later make seing steps and diffs a defult and print them with each step!



if __name__ == '__main__':
    pass
