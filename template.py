import sys
import math
from bisect import bisect_left
from collections import deque, Counter
from heapq import heappush, heappop
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

# Быстрое возведение в степень
def binpow(a, b, mod):
    res = 1
    a %= mod
    while b:
        if b % 2: res = res * a % mod
        a = a * a % mod
        b //= 2
    return res

# GCD и LCM
def gcd(a, b):
    while b: a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

# Префиксные суммы
def prefix_sum(arr):
    ps = [0]
    for x in arr:
        ps.append(ps[-1] + x)
    return ps

# Бинарный поиск по ответу
def bin_search(ok, ng, check):
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        if check(mid):
            ok = mid
        else:
            ng = mid
    return ok

# DFS — поиск в глубину
def dfs(v, graph, visited):
    visited[v] = True
    for u in graph[v]:
        if not visited[u]:
            dfs(u, graph, visited)

# BFS — поиск в ширину
def bfs(start, graph):
    n = len(graph)
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        v = q.popleft()
        for u in graph[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist

# Алгоритм Дейкстры — кратчайшие пути в графе с неотрицательными весами
def dijkstra(start, graph, n):
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, v = heappop(pq)
        if d > dist[v]: continue
        for u, w in graph[v]:
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                heappush(pq, (dist[u], u))
    return dist

# Z-функция — длины совпадений с префиксом
def z_function(s):
    n = len(s)
    z = [0] * n
    l = r = 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    return z

# Префикс-функция (КМП) — длины префиксов == суффиксам
def prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

# Сжатие координат — уменьшение больших значений до индексов
def compress(arr):
    b = sorted(set(arr))
    return [bisect_left(b, x) for x in arr]

# DSU (система непересекающихся множеств)
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, v):
        if v != self.p[v]:
            self.p[v] = self.find(self.p[v])
        return self.p[v]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.r[a] < self.r[b]: a, b = b, a
            self.p[b] = a
            if self.r[a] == self.r[b]: self.r[a] += 1

# Fenwick Tree (Binary Indexed Tree) — суммы на отрезке
class Fenwick:
    def __init__(self, n):
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i, x):
        i += 1
        while i <= self.n:
            self.t[i] += x
            i += i & -i

    def sum(self, i):
        i += 1
        res = 0
        while i > 0:
            res += self.t[i]
            i -= i & -i
        return res

    def range_sum(self, l, r):
        return self.sum(r) - self.sum(l - 1)

# Segment Tree — минимум/максимум/сумма на отрезке
class SegmentTree:
    def __init__(self, data):
        self.N = len(data)
        self.size = 1
        while self.size < self.N:
            self.size <<= 1
        self.tree = [float('-inf')] * (2 * self.size)
        for i in range(self.N):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, pos, value):
        pos += self.size
        self.tree[pos] = value
        while pos > 1:
            pos >>= 1
            self.tree[pos] = max(self.tree[2 * pos], self.tree[2 * pos + 1])

    def query(self, l, r):
        res = float('-inf')
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l //= 2
            r //= 2
        return res
