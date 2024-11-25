from multiprocessing import Pool
import multiprocessing 
import numpy as np
from math import ceil, log2, floor
import gmpy2

class VES(object):
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

class ParallelVES(VES):
    def __init__(self, ptsize, addops, valuesize, vectorsize, num_processes=None):
        super().__init__(ptsize, addops, valuesize, vectorsize)
        try:
            self.num_processes = num_processes or multiprocessing.cpu_count()
        except Exception as e:
            print(f"Warning: Failed to determine CPU count due to {e}. Using default value 1.")
            self.num_processes = 1

    def encode_parallel(self, V):
        """Encode a vector to a smaller size vector using parallel processing"""
        chunk_size = int(np.ceil(len(V) / self.num_processes))
        sub_vectors = [V[i:i + chunk_size] for i in range(0, len(V), chunk_size)]

        with Pool(processes=self.num_processes) as pool:
            encoded_sub_vectors = pool.map(self._encode_sub_vector, sub_vectors)

        E = [item for sublist in encoded_sub_vectors for item in sublist]
        return E

    def decode_parallel(self, E):
        """Decode a vector back to original size vector using parallel processing"""
        with Pool(processes=self.num_processes) as pool:
            decoded_sub_vectors = pool.map(self._decode_sub_vector, E)

        V = [item for sublist in decoded_sub_vectors for item in sublist]
        return V

    def _encode_sub_vector(self, sub_vector):
        """Helper function to encode a sub-vector"""
        bs = self.compratio
        e = []
        E = []
        for v in sub_vector:
            e.append(v)
            bs -= 1
            if bs == 0:
                E.append(self._batch(e))
                e = []
                bs = self.compratio
        if e:
            E.append(self._batch(e))
        return E

    def _decode_sub_vector(self, sub_E):
        """Helper function to decode a sub-vector"""
        V = []
        for e in sub_E:
            for v in self._debatch(e):
                V.append(v)
        return V