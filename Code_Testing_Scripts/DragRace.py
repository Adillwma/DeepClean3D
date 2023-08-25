import timeit                                   # Used in timer_function to test code execution times
import numpy as np                              # Numerical processing library
import matplotlib.pyplot as plt                 # Matplotlib plotting library

###Helper functions 
def drag_race(number, repeats, functions, args_lists, sig_figs=3, share_args=False):
    """
    Runs timeit.repeat on each function in the the input list 'functions' a specified number of times and 
    prints the minimum runtime for each func
    
    # Arguments:
    number:     Number of times to run the functions per repeat.
    repeats:    Number of times to time the function (each time function is timed it is run 'number' times).
    functions:  The functions to be timed, in format [function_name_1, function_name_2].
    args_lists: Arguments to pass to the functions using format [[F1-arg1, F1-arg2], [F2-arg1, F2-arg2, F2-arg3]] Unless all 
                functions take same arguments in which case pass [[shared_arg1, shared_arg2]] and then also set share_args=True.
    sig_figs:   Sets the number of significant figures for the printed results readout [Default=3].
    share_args: If all functions share the same argumnets then passing share_args=True allows user to only input them once and they are used for all fucntions [Default=False].
    
    # Returns:
    No values are returned instead function automatically prints statment with function names and min runtimes.
    """
    
    if share_args == True:
        args_lists = args_lists * len(functions)  # If share args is used the single set of arguments is copied for the numebr of function requiring them
        
    for i, function in enumerate(functions):
        
        run_times = timeit.repeat(lambda: function(*args_lists[i]), number=number, repeat=repeats)
        min_time = min(run_times)/number

        print("\nFunction: {}\nRuntime: {} ms (minimum result over {} runs)".format(function.__name__, round(min_time*1000, sig_figs), repeats))