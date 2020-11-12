from typing import List, Any


class Thread:
    def __init__(self, tid: int):
        self.tid = tid

    def run(self):
        raise NotImplementedError()

    @staticmethod
    def create(ThreadClass, num_threads: int, *args: Any) -> List['Thread']:
        return [ThreadClass(i, *args) for i in range(num_threads)]


class SharedMemory:
    def __init__(self, size: int):
        self.size = size
        self.access_pattern = [0xffff] * size

    def access(self, thread: Thread, i: int):
        self.access_pattern[i] = thread.tid

    def clear_access(self):
        self.access_pattern = [0xffff] * self.size

    def warp_bank_access(self, warp_idx: int) -> List[int]:
        return sorted([i % 32 for i, t in filter(lambda x: x[1] // 32 == warp_idx,
                                          enumerate(self.access_pattern))])


class Indexor:
    def __init__(self, PACKED: int, BANKS: int, PAGES: int, STRIDE: int,
                 SIZE: int):
        self.PACKED = PACKED
        self.BANKS = BANKS
        self.PAGES = PAGES
        self.STRIDE = STRIDE
        self.SIZE = SIZE

    def get(self, page: int, i: int, j: int) -> int:
        return page * SIZE + i * STRIDE + j // PACKED + (i * STRIDE // BANKS)


class MyThread(Thread):
    def __init__(self, tid: int, shared_mem: SharedMemory, indexor: Indexor,
                 ROWS: int, COLUMNS: int, trans: bool):
        super().__init__(tid)
        self.shared_mem = shared_mem
        self.indexor = indexor
        self.ROWS = ROWS
        self.COLUMNS = COLUMNS
        self.trans = trans
        self.page = 0

    def run(self):
        x = self.tid % (self.ROWS if self.trans else self.COLUMNS)
        y = self.tid // (self.ROWS if self.trans else self.COLUMNS)
        self.shared_mem.access(self, self.indexor.get(
            self.page, x if self.trans else y, y if self.trans else x))
        '''
        lane_idx = self.tid % 32
        warp_idx = self.tid // 32
        self.shared_mem.access(self, self.indexor.get(
            self.page, warp_idx * 4 + warp_idx % 4, lane_idx // 4
        ))

        '''
        self.page = 0 if self.page == 1 else 1


if __name__ == '__main__':
    PACKED = 1
    BANKS = 32

    PAGES = 2
    ROWS, COLUMNS = 32, 8

    THREADS = ROWS * COLUMNS // PACKED
    WARPS = THREADS // 32

    STRIDE = COLUMNS // PACKED
    SKEWS = ROWS * STRIDE // BANKS

    SIZE = (ROWS * STRIDE + SKEWS + 32 - 1) // 32 * 32

    mem = SharedMemory(PAGES * SIZE)

    indexor = Indexor(PACKED, BANKS, PAGES, STRIDE, SIZE)
    threads = MyThread.create(
        MyThread, THREADS, mem, indexor, ROWS, COLUMNS, False)

    for t in threads:
        t.run()

    for wid in range(WARPS):
        print(set(mem.warp_bank_access(wid)))

    mem.clear_access()
    print('=' * 200)

    for t in threads:
        t.run()

    for wid in range(WARPS):
        print(set(mem.warp_bank_access(wid)))
