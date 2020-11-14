import numpy as np
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
    def __init__(self, tid: int, shared_mem: SharedMemory, trans: bool):
        super().__init__(tid)
        self.shared_mem = shared_mem
        self.trans = trans

    def run(self):
        i = self.tid // 8
        j = self.tid % 8 * 4
        k = 0

        self.shared_mem.access(
            self,
            (i * (32 + 1) + (j + k)) if self.trans else ((j + k) * (32 + 1) + i)
        )


if __name__ == '__main__':
    mem = SharedMemory(32 * 33)
    threads = MyThread.create(MyThread, 256, mem, True)

    for t in threads:
        t.run()

    for wid in range(8):
        print(set(mem.warp_bank_access(wid)))
