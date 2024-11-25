from math import ceil, log2, floor
import gmpy2
import random
import time
from joblib import Parallel,delayed
import multiprocessing


class VES(object):
    def __init__(self, ptsize, addops, valuesize, vectorsize) -> None:
        super().__init__()
        self.ptsize = ptsize
        self.addops = addops
        self.valuesize = valuesize
        self.vectorsize = vectorsize
        self.elementsize = valuesize + ceil(log2(addops))
        self.compratio = floor(ptsize / self.elementsize)
        self.numbatches = ceil(self.vectorsize / self.compratio)

    def encode(self, V):
        # 原始的串行编码方法
        bs = self.compratio
        e = []
        E = []
        for v in V:
            e.append(v)
            bs -= 1
            if bs == 0:
                E.append(self._batch(e))
                e = []
                bs = self.compratio
        if e:
            E.append(self._batch(e))
        return E

    def encode_parallel(self, V):
        # 使用 joblib 的并行编码方法
        batch_size = self.compratio

        # 分割数据
        batches = [V[i:i + batch_size] for i in range(0, len(V), batch_size)]

        # 使用 Parallel 和 delayed 并行处理
        results = (Parallel(n_jobs=multiprocessing.cpu_count())
                   (delayed(self._encode_batch)(batch) for batch in batches))

        # 合并结果,展平操作
        return [item for sublist in results for item in sublist]

    def _encode_batch(self, batch):
        # 处理单个批次的数据
        e = []
        E = []
        for v in batch:
            e.append(v)
            if len(e) == self.compratio:
                E.append(self._batch(e))
                e = []
        if e:
            E.append(self._batch(e))
        return E

    def decode(self, E):
        """decode a vector back to original size vector"""
        V = []
        for e in E:
            for v in self._debatch(e):
                V.append(v)
        return V

    def _batch(self, V):
        i = 0
        a = 0
        for v in V:
            a |= v << self.elementsize*i
            i += 1
        return gmpy2.mpz(a)

    def _debatch(self, b):
        i = 1
        V = []
        bit = 0b1
        mask = 0b1
        for _ in range(self.elementsize-1):
            mask <<= 1
            mask |= bit

        while b != 0:
            v = mask & b
            V.append(int(v))
            b >>= self.elementsize
        return V


if __name__ == '__main__':
    keysize = 2048
    nclients = 100
    valuesize = 32
    vectorsize = 10000
    VE = VES(keysize, nclients, valuesize, vectorsize)
    vector = [random.randint(0, 2 ** 32 - 1) for _ in range(vectorsize)]

    # encode the vector
    start_time = time.perf_counter()
    E = VE.encode_parallel(V=vector)
    end_time = time.perf_counter()
    run_time_ms = (end_time - start_time) * 1000
    print(multiprocessing.cpu_count())
    print(f"程序运行时间为: {run_time_ms:.6f} 毫秒")

    print("encoded decreased the vector size from {} to {}".format(len(vector), len(E)))

    # decode the vector
    V = VE.decode(E=E)
    print("Verify:", V == vector)