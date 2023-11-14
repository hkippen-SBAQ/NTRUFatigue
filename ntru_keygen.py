# Code adapted from:
# https://github.com/kpatsakis/NTRU_Sage/blob/master/ntru.sage
# Under GPL 2.0 license.
from numpy import array, zeros, identity, block
from scipy.linalg import circulant
from numpy.random import shuffle
from numpy.random import choice as np_random_choice
from numpy import random
import numpy as np

from math import factorial as fac
from math import ceil, erf, sqrt, exp


def build_Gaussian_law(sigma, t):
    D = {}
    for i in range(0, t + 1):
        D[i] = exp(-i ** 2 / (2 * sigma ** 2))
        D[-i] = D[i]
    normalization = sum([D[i] for i in D])
    for i in D:
        D[i] = D[i] / normalization
    assert abs(sum([D[i] for i in range(-t, t + 1)]) - 1.) <= 10 ** -10
    return D


def gaussian_center_weight(sigma, t):
    """ Weight of the gaussian of std deviation s, on the interval [-t, t]
    :param x: (float)
    :param y: (float)
    :returns: erf( t / (sigma*sqrt 2) )
    """
    return erf(t / (sigma * sqrt(2.)))


def binomial(x, y):
    """ Binomial coefficient
    :param x: (integer)
    :param y: (integer)
    :returns: y choose x
    """
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom


def centered_binomial_pdf(k, x):
    """ Probability density function of the centered binomial law of param k at x
    :param k: (integer)
    :param x: (integer)
    :returns: p_k(x)
    """
    return binomial(2 * k, x + k) / 2.**(2 * k)


def build_centered_binomial_law(k):
    """ Construct the binomial law as a dictionnary
    :param k: (integer)
    :param x: (integer)
    :returns: A dictionnary {x:p_k(x) for x in {-k..k}}
    """
    D = {}
    for i in range(-k, k + 1):
        D[i] = centered_binomial_pdf(k, i)
    return D


def build_uniform_law(p):
    """ Construct the binomial law as a dictionnary
    :param k: (integer)
    :param x: (integer)
    :returns: A dictionnary {x:p_k(x) for x in {-k..k}}
    """
    D = {}
    for i in range(p):
        D[i - p // 2] = 1. / p
    return D


def mod_switch(x, q, rq):
    """ Modulus switching (rounding to a different discretization of the Torus)
    :param x: value to round (integer)
    :param q: input modulus (integer)
    :param rq: output modulus (integer)
    """
    return int(round(1. * rq * x / q) % rq)


def mod_centered(x, q):
    """ reduction mod q, centered (ie represented in -q/2 .. q/2)
    :param x: value to round (integer)
    :param q: input modulus (integer)
    """
    a = x % q
    if a < q / 2:
        return a
    return a - q


def build_mod_switching_error_law(q, rq):
    """ Construct Error law: law of the difference
    introduced by switching from and back a uniform value mod q
    :param q: original modulus (integer)
    :param rq: intermediate modulus (integer)
    """
    D = {}
    V = {}
    for x in range(q):
        y = mod_switch(x, q, rq)
        z = mod_switch(y, rq, q)
        d = mod_centered(x - z, q)
        D[d] = D.get(d, 0) + 1. / q
        V[y] = V.get(y, 0) + 1

    return D


def law_convolution(A, B):
    """ Construct the convolution of two laws
    (sum of independent variables from two input laws)
    :param A: first input law (dictionnary)
    :param B: second input law (dictionnary)
    """

    C = {}
    for a in A:
        for b in B:
            c = a + b
            C[c] = C.get(c, 0) + A[a] * B[b]
    return C


def law_product(A, B):
    """ Construct the law of the product of independent
    variables from two input laws
    :param A: first input law (dictionnary)
    :param B: second input law (dictionnary)
    """
    C = {}
    for a in A:
        for b in B:
            c = a * b
            C[c] = C.get(c, 0) + A[a] * B[b]
    return C


def clean_dist(A):
    """ Clean a distribution to accelerate furthe
     computation (drop element of the support
     with proba less than 2^-300)
    :param A: input law (dictionnary)
    """
    B = {}
    for (x, y) in A.items():
        if y > 2**(-300):
            B[x] = y
    return B

def renormalize_dist(A):
    B = {}
    summ = sum([y for (x,y) in A.items()])
    for x in A:
        B[x] = A[x] / summ 
    return B


def iter_law_convolution(A, i):
    """ compute the -ith forld convolution of a distribution (using double-and-add)
    :param A: first input law (dictionnary)
    :param i: (integer)
    """
    D = {0: 1.0}
    i_bin = bin(i)[2:]  # binary representation of n
    for ch in i_bin:
        D = law_convolution(D, D)
        D = clean_dist(D)
        if ch == '1':
            D = law_convolution(D, A)
            D = clean_dist(D)
    return D


def tail_probability(D, t):
    '''
    Probability that an drawn from D is strictly greater than t in absolute value
    :param D: Law (Dictionnary)
    :param t: tail parameter (integer)
    '''
    s = 0
    ma = max(D.keys())
    if t >= ma:
        return 0
    # Summing in reverse for better numerical precision (assuming tails are decreasing)
    for i in reversed(range(int(ceil(t)), ma)):
        s += D.get(i, 0) + D.get(-i, 0)
    return s

def draw_from_distribution(D, shape):
    """draw an element from the distribution D
    :D: distribution in a dictionnary form
    """
    X = np_random_choice([key for key in D.keys()],
                         1, replace=True,
                         size=shape,
                         p=[float(prob) for prob in D.values()])
    return X

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise ZeroDivisionError
    else:
        return x % m


def modinvMat(M, q):
    n, m = M.shape
    assert m==n

    invs = q * [None]
    for i in range(1, q):
        try:
            invs[i] = modinv(i, q)
            assert((i*invs[i]) % q == 1)
        except ZeroDivisionError:
            pass

    R = block([[M, identity(n, dtype="long")]])
    #print(R)

    # i-th column
    for i in range(n):
        #print(i, q, R)
        # Find a row with an invertible i-th coordinate
        for j in range(i, n+1):
            if j == n:
                raise ZeroDivisionError

            # Normalize the row and swap it with row j
            if invs[R[j,i]] is not None:
                R[j] = (R[j] * invs[R[j,i]]) % q

                if j > i:
                    R[i], R[j] = R[j], R[i]
                break

        # Kill all coordinates of that column except at row j
        for j in range(n):
            if i==j: continue
            R[j] = (R[j] -  R[i] * R[j, i]) % q

    #print(i, R)

    Minv = R[:,n:]
    return Minv


def DiscreteGaussian(shape, sigmasq):
    sz = int(np.ceil(10*np.sqrt(sigmasq)))
    interval = range(-sz, sz+1)
    p = [np.exp(-x*x/(2*sigmasq)) for x in interval]
    p /= np.sum(p)
    return np.random.choice(interval, shape, p=p)


def sample_distribution(dist, **kwargs):
    if dist.strip().lower() == "discrete_gaussian":  # param_1 = variance
        return DiscreteGaussian(shape=kwargs["shape"], sigmasq=kwargs["dist_param_1"])

    elif dist.strip().lower() == "binomial":  # param_1 = range
        D = build_centered_binomial_law(kwargs["dist_param_1"])
        return np.array(draw_from_distribution(D, kwargs["shape"]))

    elif dist.strip().lower() == "uniform":  # param_1 = uniform modulus
        D = build_uniform_law(kwargs["dist_param_1"])
        return np.array(draw_from_distribution(D, kwargs["shape"]))

    elif dist.strip().lower() == "sparse_ternary":  # param_1 = sparsity
        D = {-1: 0.5, 1: 0.5}
        hamming_weight = round(kwargs["shape"]*kwargs["dist_param_1"])
        return shuffle(np.array([draw_from_distribution(D, 1)[0] for _ in range(hamming_weight)]
                                + (kwargs["shape"] - hamming_weight)*[0]))


class NTRUEncrypt_Matrix:

    def gen_keys(self):
        while True:
            F = sample_distribution(self.dist, shape=(self.n,self.n), dist_param_1=self.dist_param_1)
            try:
                Finv = modinvMat(F, self.q)
                break
            except ZeroDivisionError:
                # print("failed inverse")
                continue
        
        G = sample_distribution(self.dist, shape=(self.n,self.n), dist_param_1=self.dist_param_1)
        H = Finv.dot(G) % self.q
        return H, F, G

    def __init__(self, n, q, dist, dist_param_1):
        self.n = n
        self.q = q
        self.dist = dist
        self.dist_param_1 = dist_param_1


class NTRUEncrypt_Circulant:

    def gen_keys(self):
        while True:
            f = sample_distribution(self.dist, shape=self.n, dist_param_1=self.dist_param_1)
            F = circulant(f)
            try:
                Finv = modinvMat(F, self.q)
                break
            except ZeroDivisionError:
                # print("failed inverse")
                continue

        g = sample_distribution(self.dist, shape=self.n, dist_param_1=self.dist_param_1)
        G = circulant(g)
        H = Finv.dot(G) % self.q
        return H, F, G

    def __init__(self, n, q, dist, dist_param_1):
        self.n = n
        self.q = q
        self.dist = dist
        self.dist_param_1 = dist_param_1

class NTRUEncrypt:

    def sample_ternary(self, ones, minus_ones):
        s = [1]*ones + [-1]*minus_ones + [0]*(self.n - ones - minus_ones)
        shuffle(s)
        return s


    def gen_keys(self):
        while True:
            f = self.sample_ternary(self.Df, self.Df-1)
            F = circulant(f)
            try:
                Finv = modinvMat(F, self.q)
                break
            except ZeroDivisionError:
                # print("failed inverse")
                continue

        g = self.sample_ternary(self.Dg, self.Dg)
        G = circulant(g)
        H = G.dot(Finv) % self.q
        return H, F, G

    def __init__(self, n, q, Df, Dg):
        self.n = n
        self.q = q
        self.q = q
        self.Df = Df
        self.Dg = Dg

def build_ntru_lattice(n, q, H):

    lambd = block([[q * identity(n, dtype="long") , zeros((n, n), dtype="long") ],
                   [     H            , identity(n, dtype="long") ] ])
    return lambd

def gen_ntru_instance_matrix(n, q, dist, dist_param_1, seed=None):
    random.seed(np.uint32(seed))
    ntru = NTRUEncrypt_Matrix(n, q, dist, dist_param_1)
    H, F, G = ntru.gen_keys()
    B = build_ntru_lattice(n, q, H)
    return B, F, G

def gen_ntru_instance_circulant(n, q, dist, dist_param_1, seed=None):
    random.seed(np.uint32(seed))
    ntru = NTRUEncrypt_Circulant(n, q, dist, dist_param_1)
    H, F, G = ntru.gen_keys()
    B = build_ntru_lattice(n, q, H)
    return B, F, G

def gen_ntru_instance(n, q, Df=None, Dg=None, seed=None):
    random.seed(np.uint32(seed))
    if Df is None:
        Df = n//3
    if Dg is None:
        Dg = n//3

    ntru = NTRUEncrypt(n, q, Dg, Df)
    H, F, G = ntru.gen_keys()
    B = build_ntru_lattice(n, q, H.transpose())
    return B, [F[0], G[0]]
