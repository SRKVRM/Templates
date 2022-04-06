# pujauzkfpltummwvlc@upived.com

# To take input from txt file and save to txt file
python program.py < input.txt > output.txt

#=====================================================

from sys import stdin, stdout
import heapq
import cProfile, math
from collections import Counter, defaultdict, deque
from bisect import bisect_left, bisect, bisect_right
import itertools
from copy import deepcopy
from fractions import Fraction
import sys, threading
import operator as op
from functools import reduce
import sys
 
sys.setrecursionlimit(10 ** 6)  # max depth of recursion
threading.stack_size(2 ** 27)  # new thread will get stack of such size
fac_warm_up = False
printHeap = str()
memory_constrained = False
P = 10 ** 9 + 7
 
 
class MergeFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.num_sets = n
        self.lista = [[_] for _ in range(n)]
 
    def find(self, a):
        to_update = []
        while a != self.parent[a]:
            to_update.append(a)
            a = self.parent[a]
        for b in to_update:
            self.parent[b] = a
        return self.parent[a]
 
    def merge(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.num_sets -= 1
        self.parent[b] = a
        self.size[a] += self.size[b]
        self.lista[a] += self.lista[b]
 
    def set_size(self, a):
        return self.size[self.find(a)]
 
    def __len__(self):
        return self.num_sets
 
 
def display(string_to_print):
    stdout.write(str(string_to_print) + "\n")
 
 
def fast_exp(base, power):
    result = 1
    while power > 0:
        if power % 2 == 1:
            result = (result * base) % m
        power = power // 2
        base = (base * base) % m
    return result
 

# n**0.5 complexity 
def prime_factors(n):  
    factors = dict()
    for i in range(2, math.ceil(math.sqrt(n)) + 1):
        while n % i == 0:
            if i in factors:
                factors[i] += 1
            else:
                factors[i] = 1
            n = n // i
    if n > 2:
        factors[n] = 1
    return (factors)
 
 
def all_factors(n):
    return set(reduce(list.__add__,([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))
 
 
def fibonacci_modP(n, MOD):
    if n < 2: return 1
    return (cached_fn(fibonacci_modP, (n + 1) // 2, MOD) * cached_fn(fibonacci_modP, n // 2, MOD) + cached_fn(fibonacci_modP, (n - 1) // 2, MOD) * cached_fn(fibonacci_modP, (n - 2) // 2, MOD)) % MOD
 
 
def factorial_modP_Wilson(n, p):
    if (p <= n):
        return 0
    res = (p - 1)
    for i in range(n + 1, p):
        res = (res * cached_fn(InverseEuler, i, p)) % p
    return res
 
 
def binary(n, digits=20):
    b = bin(n)[2:]
    b = '0' * (digits - len(b)) + b
    return b
 
 
def is_prime(n):
    """Returns True if n is prime."""
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return False
    i = 5
    w = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += w
        w = 6 - w
    return True
 
#  Sieve of Eratosthenes 
def sieve(n):
    prime = [True for i in range(n + 1)]
    p = 2
    while p * p <= n:
        if prime[p]:
            for i in range(p * 2, n + 1, p):
                prime[i] = False
        p += 1
    return prime
    
#  O(nlog(logn))
 


factorial_modP = []
 
 
def warm_up_fac(MOD):
    global factorial_modP, fac_warm_up
    if fac_warm_up: return
    factorial_modP = [1 for _ in range(fac_warm_up_size + 1)]
    for i in range(2, fac_warm_up_size):
        factorial_modP[i] = (factorial_modP[i - 1] * i) % MOD
    fac_warm_up = True
 
 
def InverseEuler(n, MOD):
    return pow(n, MOD - 2, MOD)

def nCk(n, k): 
    if(k > n - k): 
        k = n - k 
    res = 1
    for i in range(k): 
        res = res * (n - i) 
        res = res / (i + 1) 
    return res 
 
def nCr(n, r, MOD):
    global fac_warm_up, factorial_modP
    if not fac_warm_up:
        warm_up_fac(MOD)
        fac_warm_up = True
    return (factorial_modP[n] * (
            (pow(factorial_modP[r], MOD - 2, MOD) * pow(factorial_modP[n - r], MOD - 2, MOD)) % MOD)) % MOD
 
 
def test_print(*args):
    if testingMode:
        print(args)
 
 
def display_list(list1, sep=" "):
    stdout.write(sep.join(map(str, list1)) + "\n")
 
 
def display_2D_list(li):
    for i in li:
        print(i)
 
 
def prefix_sum(li):
    sm = 0
    res = []
    for i in li:
        sm += i
        res.append(sm)
    return res
 
 
def get_int():
    return int(stdin.readline().strip())
 
 
def get_tuple():
    return map(int, stdin.readline().split())
 
 
def get_list():
    return list(map(int, stdin.readline().split()))
 
 
memory = dict()
 
 
def clear_cache():
    global memory
    memory = dict()
 
 
def cached_fn(fn, *args):
    global memory
    if args in memory:
        return memory[args]
    else:
        result = fn(*args)
        memory[args] = result
        return result
 
 
def ncr(n, r):
    return math.factorial(n) / (math.factorial(n - r) * math.factorial(r))
 
 
def binary_search(i, li):
    fn = lambda x: li[x] - x // i
    x = -1
    b = len(li)
    while b >= 1:
        while b + x < len(li) and fn(b + x) > 0:  # Change this condition 2 to whatever you like
            x += b
        b = b // 2
    return x
 
 
 
 
 
exit()
 
 
####################################################################################



from __future__ import division, print_function

''' Hey stalker :) '''
INF = 10 ** 10
TEST_CASES = True
from collections import defaultdict, Counter


def main():
    
    #
    #
    #
    #    MAIN CODE HERE
    #
    #
    #
    #
    #
    pass

''' FastIO Footer: PyRival Library, Thanks @c1729 and contributors '''
import os
import sys
from bisect import bisect_left, bisect_right
from io import BytesIO, IOBase

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip
BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")


def print(*args, **kwargs):
    """Prints the values to a stream, or to sys.stdout by default."""
    sep, file = kwargs.pop("sep", " "), kwargs.pop("file", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop("end", "\n"))
    if kwargs.pop("flush", False):
        file.flush()


if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
# else:
#     sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip("\r\n")
get_int = lambda: int(input())
get_list = lambda: list(map(int, input().split()))
if __name__ == "__main__":
    if TEST_CASES:
        [main() for _ in range(int(input()))]
    else:
        main()




