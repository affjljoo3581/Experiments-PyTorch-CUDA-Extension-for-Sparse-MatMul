import numpy as np

PACKED = 2
ROWS, COLUMNS = 32, 8
trans = True

THREADS = ROWS * COLUMNS // PACKED
tid = np.arange(THREADS)


xs = tid * PACKED % (ROWS if trans else COLUMNS)
ys = tid * PACKED // (ROWS if trans else COLUMNS)

memory = np.zeros((COLUMNS if trans else ROWS, ROWS if trans else COLUMNS))
for t, x, y in zip(tid, xs, ys):
    memory[y, x] = t

print(memory)
