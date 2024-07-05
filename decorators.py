def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{method.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper