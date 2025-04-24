#include <bits/stdc++.h>
using namespace std;

// =================== Макросы и типы ===================
#define int long long
#define pii pair<int, int>
#define vi vector<int>
#define vvi vector<vi>
#define all(x) (x).begin(), (x).end()
#define sz(x) ((int)(x).size())
#define pb push_back
#define ff first
#define ss second

void fast_io() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
}

// =================== Математика ===================
int gcd(int a, int b) { return b ? gcd(b, a % b) : a; }
int lcm(int a, int b) { return a / gcd(a, b) * b; }

int binpow(int a, int b, int m) {
    int res = 1;
    a %= m;
    while (b) {
        if (b & 1) res = res * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return res;
}

// =================== Бинарный поиск ===================
int bin_search(int l, int r, function<bool(int)> check) {
    while (l < r) {
        int m = (l + r) / 2;
        if (check(m)) r = m;
        else l = m + 1;
    }
    return l;
}

// =================== Coordinate Compression ===================
vi compress(vi& a) {
    vi b = a;
    sort(all(b));
    b.erase(unique(all(b)), b.end());
    vi res;
    for (int x : a)
        res.pb(lower_bound(all(b), x) - b.begin());
    return res;
}

// =================== Префиксные суммы ===================
vi prefix_sum(const vi& a) {
    vi p(sz(a) + 1);
    for (int i = 0; i < sz(a); ++i)
        p[i + 1] = p[i] + a[i];
    return p;
}

// =================== Графы ===================
void dfs(int v, vvi& g, vector<bool>& used) {
    used[v] = true;
    for (int u : g[v])
        if (!used[u]) dfs(u, g, used);
}

void bfs(int s, vvi& g, vi& dist) {
    queue<int> q;
    q.push(s);
    dist[s] = 0;
    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (int u : g[v]) {
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                q.push(u);
            }
        }
    }
}

void dijkstra(int s, vvi& g, vi& dist) {
    int n = sz(g);
    dist.assign(n, 1e18);
    dist[s] = 0;
    priority_queue<pii, vector<pii>, greater<>> pq;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, v] = pq.top(); pq.pop();
        if (d > dist[v]) continue;
        for (int u : g[v]) {
            if (dist[v] + 1 < dist[u]) {
                dist[u] = dist[v] + 1;
                pq.push({dist[u], u});
            }
        }
    }
}

// =================== Fenwick Tree ===================
struct Fenwick {
    int n;
    vi bit;
    Fenwick(int n) : n(n), bit(n, 0) {}

    void add(int i, int x) {
        for (; i < n; i |= (i + 1))
            bit[i] += x;
    }

    int sum(int r) {
        int res = 0;
        for (; r >= 0; r = (r & (r + 1)) - 1)
            res += bit[r];
        return res;
    }

    int sum(int l, int r) {
        return sum(r) - sum(l - 1);
    }
};

// =================== Segment Tree ===================
struct SegTree {
    int n;
    vi t;
    SegTree(int n) : n(n), t(4 * n, 1e18) {}

    void update(int v, int tl, int tr, int pos, int val) {
        if (tl == tr) t[v] = val;
        else {
            int tm = (tl + tr) / 2;
            if (pos <= tm) update(v * 2, tl, tm, pos, val);
            else update(v * 2 + 1, tm + 1, tr, pos, val);
            t[v] = min(t[v * 2], t[v * 2 + 1]);
        }
    }

    int get_min(int v, int tl, int tr, int l, int r) {
        if (l > r) return 1e18;
        if (l == tl && r == tr) return t[v];
        int tm = (tl + tr) / 2;
        return min(
            get_min(v * 2, tl, tm, l, min(r, tm)),
            get_min(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r)
        );
    }
};

// =================== Подсчёт инверсий ===================
int count_inversions(vi a) {
    int n = sz(a);
    vi b = compress(a);
    Fenwick fw(n);
    int inv = 0;
    for (int i = n - 1; i >= 0; --i) {
        inv += fw.sum(b[i] - 1);
        fw.add(b[i], 1);
    }
    return inv;
}

// =================== Утилиты ===================
template<typename T>
void print_vec(const vector<T>& v) {
    for (auto x : v) cout << x << ' ';
    cout << '\n';
}

// =================== main ===================
int32_t main() {
    fast_io();

    // Пример использования
    int n;
    cin >> n;
    vi a(n);
    for (int& x : a) cin >> x;
    vi psum = prefix_sum(a);
    print_vec(psum);

    return 0;
}
