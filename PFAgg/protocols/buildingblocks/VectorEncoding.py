"""
### ** Vector Encoding Scheme **

This module encodes and decodes vector elements to create smaller size vectors using a packing technique
"""
"""
The vector encoding class

** Args**:
-------------
*ptsize* : `int` --
    The bitlength of the plaintext (the number of bits of an element in the output vector) 

*addops* : `int` --
    The number of supported addition operation on the encoded elements (to avoid overflow)

*valuesize* : `int` --
    The bit length of an element of the input vector

*vectorsize* : `int` --
    The number of element of the input vector

** Attributes**:
-------------
*ptsize* : `int` --
    The bitlength of the plaintext (the number of bits of an element in the output vector) 

*addops* : `int` --
    The number of supported addition operation on the encoded elements (to avoid overflow)

*valuesize* : `int` --
    The bit length of an element of the input vector

*vectorsize* : `int` --
    The number of element of the input vector

*elementsize* : `int` --
    The extended bit length of an element of the input vector

*compratio* : `int` --
    The compression ratio of the scheme

*numbatches* : `int` --
    The number of elements in the output vector


"""

from math import ceil, log2, floor
import gmpy2
import random
import time

class VES(object):


    def __init__(self, ptsize, addops, valuesize, vectorsize) -> None:
        super().__init__()
        self.ptsize = ptsize
        self.addops = addops
        self.valuesize = valuesize
        self.vectorsize = vectorsize
        self.elementsize = valuesize + ceil(log2(addops))
        # self.elementsize = valuesize + ceil(log2(addops + 1))
        self.compratio = floor(ptsize / self.elementsize)
        self.numbatches = ceil(self.vectorsize / self.compratio)

    def encode(self, V):
        """Encode a vector to a smaller size vector"""
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
    E = VE.encode(V=vector)
    end_time = time.perf_counter()
    run_time_ms = (end_time - start_time) * 1000
    print(f"程序运行时间为: {run_time_ms:.6f} 毫秒")

    print("encoded decreased the vector size from {} to {}".format(len(vector), len(E)))

    # decode the vector
    V = VE.decode(E=E)
    print("Verify:", V == vector)