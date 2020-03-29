import logging

from time import perf_counter

logger = logging.getLogger(__name__)


def time_counter(func):
    if logger.level == logging.DEBUG:
        def wrapper(*args, **kwargs):
            start = perf_counter()
            res = func(*args, **kwargs)
            print(f'Time for {func.__name__} = {(perf_counter() - start):.2f} sec.')
            print('*' * 10)
            return res

        return wrapper
    else:
        return func
