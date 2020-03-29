from time import perf_counter


def time_counter(func):

    def wrapper(*args, **kwargs):
        start = perf_counter()
        res = func(*args, **kwargs)
        print(f'Time for {func.__name__} = {(perf_counter() - start):.2f} sec.')
        print('*' * 10)
        return res

    return wrapper
