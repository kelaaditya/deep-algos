def batchify(X, y, batch_size):
    for i in range(0, len(X) - batch_size + 1, batch_size):
        yield(X[i: i + batch_size], y[i: i + batch_size])