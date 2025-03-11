from functools import wraps
from time import perf_counter
from datetime import date

def timer(function):
    @wraps(function)
    def new_function(*args, **kwargs):
        start_time = perf_counter()
        result = function(*args)
        elapsed = perf_counter() - start_time
        print('\nFunction "{name}" took {time} seconds to complete.\n'.format(
            name=function.__name__, time=elapsed))
        return result

    return new_function