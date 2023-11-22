from math import factorial as fac
from math import ceil, erf, sqrt, exp
from numpy.random import choice as np_random_choice
from numpy.random import shuffle
import numpy as np

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
        D[i - p / 2] = 1. / p
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
                         size=shape, replace=True,
                         p=[float(prob) for prob in D.values()])
    return X


def average_variance(D):
    mu = 0.
    s = 0.

    for (v, p) in D.items():
        mu += v * p
        s += v * v * p

    s -= mu * mu
    return mu, s


class Probability_Distribution:

    def __init__(self, dist_name, param, dim=None):
        self.dist_name = dist_name.strip().lower()
        self.param = param
        self.dim = dim

        if self.dist_name == "discrete_gaussian":  # param_1 = variance
            self.D = build_Gaussian_law(np.sqrt(self.param), round(10*np.sqrt(self.param)))

        elif self.dist_name == "binomial":  # param_1 = range
            self.D = build_centered_binomial_law(self.param)

        elif self.dist_name == "uniform":  # param_1 = uniform modulus
            self.D = build_uniform_law(self.param)

        elif self.dist_name == "sparse_ternary":
            if self.param < 1:
                self.param = round(self.param*self.dim)

            self.D = None

        else:
            self.D = None

    def sample_distribution(self, shape):
        if self.D is not None:
            return np.array(draw_from_distribution(self.D, shape))

        elif self.D is None and self.dist_name == "sparse_ternary":  # param_1 = sparsity
            D = {-1: 0.5, 1: 0.5}
            try:
                n, m = shape
                arr = np.array([draw_from_distribution(D, 1)[0] for _ in range(self.param)]
                                    + ((n*m) - self.param)*[0]).reshape(shape)
                shuffle(arr)
                return arr

            except:
                arr = np.array([draw_from_distribution(D, 1)[0] for _ in range(self.param)]
                               + (shape - self.param)*[0])
                shuffle(arr)
                return arr

        else:
            raise NotImplementedError(f"Distribution: {self.dist_name} not supported!")

    @property
    def stddev(self):
        if self.D is not None:
            _, variance = average_variance(self.D)
            return np.sqrt(variance)

        elif self.D is None and self.dist_name == "sparse_ternary":
            return np.sqrt(self.param/self.dim)

        else:
            raise NotImplementedError(f"Distribution: {self.dist_name} not supported!")
