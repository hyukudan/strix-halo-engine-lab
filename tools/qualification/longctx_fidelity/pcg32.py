"""PCG XSH RR 64/32, independently expressed from its arithmetic specification.

64-bit LCG multiplier 6364136223846793005, increment (stream<<1)|1;
output xorshift ((old>>18)^old)>>27 truncated to 32 bits, rotate old>>59.
Standard two-step seeding: state=0; step; add seed modulo 2**64; step.
Default stream=54. All bounds use rejection, never Python random or hash().
"""
MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
MULTIPLIER = 6364136223846793005


class PCG32:
    def __init__(self, seed, stream=54):
        if not 0 <= seed <= MASK64 or not 0 <= stream < (1 << 63):
            raise ValueError("seed/stream out of range")
        self.state = 0
        self.increment = (stream << 1) | 1
        self.next()
        self.state = (self.state + seed) & MASK64
        self.next()

    def next(self):
        old = self.state
        self.state = (old * MULTIPLIER + self.increment) & MASK64
        word = (((old >> 18) ^ old) >> 27) & MASK32
        rotation = old >> 59
        return ((word >> rotation) | (word << ((-rotation) & 31))) & MASK32

    def below(self, bound):
        if not 1 <= bound <= (1 << 32):
            raise ValueError("invalid bound")
        threshold = (1 << 32) % bound
        while True:
            value = self.next()
            if value >= threshold:
                return value % bound

    def choice(self, values):
        return values[self.below(len(values))]

    def opaque(self):
        return "".join(f"{self.next():08x}" for _ in range(4))
