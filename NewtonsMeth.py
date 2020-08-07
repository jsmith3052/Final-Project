import numpy as np

def deriv(x, func): #works for analytic functions !
    h = 1e-5
    return (func(x+h)-func(x-h))/(2*h) #using the symmetric derivative
       
def sin(x):
    return np.sin(x)

def d_sin(x):
    return np.cos(x)

def comp_sol(x):
    return x**3 - 1

def d_com_sol(x):
    return (3*x)**2

def real_sol(x):
    return np.exp(x) - 5*x

def d_real_sol(x):
    return np.exp(x) - 5



def newtons_method(func, d_func, x0, Print = False , step_lim= 10, tol= 1e-10): ###default parameter to see column info######
    """takes in the function, manyually computed deriv, starting point
        step_lim is how many iterations you want the method to take
        tol is how exact you want your error of the sol to be (distance from zero),
            note we use this tol value elsewhere where it is needed
        
        The function returns the final step value, and arrays of the difference 
            steps and other relevant information
        If Print == True the you can see the step and diff to the next step at every iteration   
        """
    
    diff = 1 #arbitraty value
    step = 0
    step_array =[]
    diff_array =[]
    while diff > tol and step < step_lim:
        step += 1
        if d_func(x0) == 0:
            print(f"function has zero derivative at {x0}")
            return
        x1 = x0 - (func(x0)/d_func(x0)) #solve for next point     
        diff = abs(x0 - x1)
        if Print == True:
            print(f"{x0:.3f} , {diff:.3f}")
        step_array.append(x0)
        diff_array.append(diff)
        x0 = x1
    if diff > tol:
        print("failure to find root \n number of steps is \n {} n With the final step of {}".format(step, x0))
        return     
    else:    
        return x0, step







if __name__ == "__main__":
    def f(x):
        return x**2 +1
    res = newtons_method(lambda x: x**2,lambda x: 2*x, 1j, Print=True)
    print(res)
#diff between next and previous point decreses by ^2 value   
#if the root cannot be found at an example where it reaches the min (where the root would be)\
    # and slope approaches 0 it changes violently. look at the steps x takes for\
    #newtons_method(lambda x: x**2 + 1,lambda x: 2*x, 5)
#Note that if the functon only has real or complex roots you must input a real or complex
    #value respectively for the solution to converge to the root
    
    
    ##Can you go to imaginary input to real solution
    
    #later make seing steps and diffs a defult and print them with each step!
    
