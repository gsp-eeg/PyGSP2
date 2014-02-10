#Filter box

import numpy as np
import matplotlib.pylab as plt

class filter:
    """A filter object"""
    def __init__(self):
        self.g = 0
        self.c_coeffs = 0

    def function(self,type='Heatkernel',param=1):
        """ store in the filter a lambda function according the
        type of filter specified.
        type : type of filter
        Heatkernel, param is decrease rate
        Gaussian, param is the variance
        Rect, param is the rectangle width
        """
        if type == 'Heatkernel':
            self.g = lambda x: np.exp(- param * x)
        elif type == 'Gaussian':
            self.g = lambda x: np.exp(- x**2/param)
        elif type=='Rect':
            self.g = lambda x: (x<param)*1.0
        else:
            print "Unknow filter type"


def cheby_coeff(g, m, N=None, arange=(-1,1)): #copy/paste of Weinstein
    """ Compute Chebyshev coefficients of given function.
        
        Parameters
        ----------
        g : function handle, should define function on arange
        m : maximum order Chebyshev coefficient to compute
        N : grid order used to compute quadrature (default is m+1)
        arange : interval of approximation (defaults to (-1,1) )
        
        Returns
        -------
        c : list of Chebyshev coefficients, ordered such that c(j+1) is
        j'th Chebyshev coefficient
        """
    if N is None:
        N = m+1
    
    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0
    n = np.pi * (np.r_[1:N+1] - 0.5) / N
    s = g(a1 * np.cos(n) + a2)
    c = np.zeros(m+1)
    for j in range(m+1):
        c[j] = np.sum(s * np.cos(j * n)) * 2 / N
    
    return c




def filtfunction(type='Heatkernel',param=1):
    """ return a lambda function according the
    type of filter specified.
    type : type of filter
        Heatkernel, param is decrease rate
        Gaussian, param is the variance
        Rect, param is the rectangle width
    Return
    g Lambda function
    """
    if type == 'Heatkernel':
        g = lambda x: np.exp(- param * x)
    elif type == 'Gaussian':
        g = lambda x: np.exp(- x**2/param)
    elif type=='Rect':
        g = lambda x: (x<param)*1.0
    else:
        print "Unknow filter type"
    return g


def cheby_op(f, L, c, arange):#copy/paste of Weinstein,must be adapted
    """Compute (possibly multiple) polynomials of laplacian (in Chebyshev
        basis) applied to input.
        
        Coefficients for multiple polynomials may be passed as a lis. This
        is equivalent to setting
        r[0] = cheby_op(f, L, c[0], arange)
        r[1] = cheby_op(f, L, c[1], arange)
        ...
        
        but is more efficient as the Chebyshev polynomials of L applied to f can be
        computed once and shared.
        
        Parameters
        ----------
        f : input vector
        L : graph laplacian (should be sparse)
        c : Chebyshev coefficients. If c is a plain array, then they are
        coefficients for a single polynomial. If c is a list, then it contains
        coefficients for multiple polynomials, such  that c[j](1+k) is k'th
        Chebyshev coefficient the j'th polynomial.
        arange : interval of approximation
        
        Returns
        -------
        r : If c is a list, r will be a list of vectors of size of f. If c is
        a plain array, r will be a vector the size of f.
        """
    if not isinstance(c, list) and not isinstance(c, tuple):
        r = cheby_op(f, L, [c], arange)
        return r[0]
    
    N_scales = len(c)
    M = np.array([coeff.size for coeff in c])
    max_M = M.max()
    
    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0
    
    Twf_old = f
    Twf_cur = (L*f - a2*f) / a1
    r = [0.5*c[j][0]*Twf_old + c[j][1]*Twf_cur for j in range(N_scales)]
    
    for k in range(1, max_M):
        Twf_new = (2/a1) * (L*Twf_cur - a2*Twf_cur) - Twf_old
        for j in range(N_scales):
            if 1 + k <= M[j] - 1:
                r[j] = r[j] + c[j][k+1] * Twf_new
        
        Twf_old = Twf_cur
        Twf_cur = Twf_new
    
    return r

def view_filter(g, c_coeffs, lambdamax): #adapted from Weinstein view_design
    """Plot the filter in the spectral domain and its Chebychev approx.
        
        Plot the input scaling function and wavelet kernels, indicates the wavelet
        scales by legend, and also shows the sum of squares G and corresponding
        frame bounds for the transform.
        
        Parameters
        ----------
        g : handle for the filter function
        c_coeffs : Chebychev coefficients of the approximation of g
        lambdamax : max spectral value, the curves are plotted on [0,lambdamax]
        
        Returns
        -------
        h : figure handle
        """
    x = np.linspace(0, lambdamax, 1e3)
    h = plt.figure()
    
    
    plt.plot(x, g(x), 'k', label='g')
    max_order =c_coeffs.size

    a1 = lambdamax / 2.0
    a2 = lambdamax / 2.0

    f=1
    Twf_old = f
    Twf_cur = (x*f - a2*f) / a1
    r = [0.5*c_coeffs[0]*Twf_old + c_coeffs[1]*Twf_cur]

    for k in range(1, max_order-1):
        Twf_new = (2/a1) * (x*Twf_cur - a2*Twf_cur) - Twf_old
        r= r + c_coeffs[k+1] * Twf_new
        Twf_old = Twf_cur
        Twf_cur = Twf_new

    plt.plot(x,r.T ,label='g_approx')
    plt.xlim(0, lambdamax)
    
    plt.title('filter kernel g in the spectral domain \n'
              'and its Chebychev approximation')
    plt.yticks(np.r_[0:4])
    plt.ylim(0, 3)
    plt.legend()
    plt.show()
    return h

if __name__ == "__main__":
    print 'This is the graph wavelet toolbox.'

