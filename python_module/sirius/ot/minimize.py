import numpy as np
from ..coefficient_array import inner


def beta_fletcher_reves(dfnext, df, P=None):
    """
    """
    if P is None:
        return np.real(inner(dfnext, dfnext) / inner(df, df))
    else:
        return np.real(inner(dfnext, P * dfnext) / inner(df, P * df))


def beta_polak_ribiere(dfnext, df, P=None):
    """
    """

    if P is None:
        return np.real(inner(dfnext, dfnext - df)) / np.real(inner(df, df))
    else:
        return np.real(inner(P * dfnext, dfnext - df) / inner(df, P * df))


def beta_sd(dfnext, df, P=None):
    """
    """
    return 0


def ls_qinterp(x0, p, f, dfx0, f0, s=0.2):
    """
    Keyword Arguments:
    x0       -- starting point
    p        -- search direction
    f        --f(x
    dfx0     -- grad f(x0)
    """
    b = np.real(2 * inner(p, dfx0))
    assert (b <= 0)
    f1 = f(x0 + s * p)
    # g(t) = a * t^2 + b * t + c
    c = f0
    a = (f1 - b * s - c) / s**2
    if a <= 1e-12:
        raise ValueError('ls_qinterp: not convex!')
    tmin = -b / (2 * a)
    x = x0 + tmin * p

    fnext = f(x)
    if not fnext < f0:
        # print('ls_qinterp failed!!!')
        # revert coefficients to x0
        f0r = f(x0)  # never remove this, need side effects
        assert (np.isclose(f0, f0r))
        raise ValueError('ls_qinterp did not improve the solution')
    return x, fnext


def gss(f, a, b, tol=1e-3):
    """
    Golden section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.


    """

    (a, b) = (min(a, b), max(a, b))

    h = b - a
    if h <= tol:
        return (a, b)

    invphi = (np.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1/phi^2
    # required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

    if yc < yd:
        # print('gss: a=%.2g, d=%.2g' % (a, d))
        return (a, d)
    else:
        # print('gss: c=%.2g, b=%.2g' % (c, b))
        return (c, b)


def ls_golden(x0, p, f, f0, **kwargs):
    """
    Golden-section search
    """
    t1, t2 = gss(lambda t: f(x0 + t * p), **kwargs, tol=1e-3)
    tmin = (t1 + t2) / 2
    x = x0 + tmin * p
    fn = f(x)
    if not fn < f0 and np.abs(fn - f0) > 1e-10:
        raise ValueError('golden-section search has failed to improve the solution')
    return x, fn


def ls_bracketing(x0, p, f, dfx0, tau=0.5, maxiter=40, **kwargs):
    """
    Bracketing line search algorithm

    Keyword Arguments:
    x0      --
    dfx0    --
    f       --
    tau     -- (default 0.5) reduction parameter
    maxiter -- (default 40)
    """
    f0 = f(x0)
    c = 0.5
    m = 2 * inner(p, dfx0)
    assert (m < 0)
    ds = 5
    assert (maxiter >= 1)

    fn = f(x0 + ds * p)
    for i in range(maxiter):
        if fn <= f0 + c * ds * m:
            break
        else:
            ds *= tau
            fn = f(x0 + ds * p)
            # print('ls_bracketing: ds: %.4g' % ds)
    else:
        if not fn < f0 and np.abs(fn - f0) > 1e-10:
            print('ls_bracketing has failded')
            raise ValueError(
                'failed to find a step length after maxiter=%d iterations.' %
                maxiter)
    return x0 + ds * p, fn


def minimize(x0,
             f,
             df,
             maxiter=100,
             tol=1e-7,
             lstype='interp',
             mtype='FR',
             M=None,
             restart=None,
             verbose=False,
             log=False):
    """
    Keyword Arguments:
    x0         -- np.narray
    f          -- function object
    df         -- function object
    maxiter    -- (default 100)
    tol        -- (default 1e-7)
    lstype     -- (default 'interp')
    mtype      -- (default 'FR')
    M          -- (default None) preconditioner
    restart    -- (default None) restart interval
    verbose    -- (default False) verbose output
    log        -- (default False) log values

    Returns: (x, iter, success)
    x          -- result
    iter       -- iteration count
    success    -- converged to specified tolerance
    [history]  -- if log==True
    """
    from ..helpers import Logger
    logger = Logger()

    if mtype == 'FR':
        beta = beta_fletcher_reves
    elif mtype == 'PR':
        beta = beta_polak_ribiere
    elif mtype == 'SD':
        beta = beta_sd
    else:
        raise Exception('invalid argument for `mtype`')

    if lstype == 'interp':
        linesearch = ls_qinterp
    elif lstype == 'golden':
        linesearch = ls_golden
    elif lstype == 'bracketing':
        linesearch = ls_bracketing
    else:
        raise Exception('invalid argument for `lstype`')

    # compute initial energy
    x = x0
    fc = f(x)
    pdfx, dfx = df(x)
    p = -M @ dfx

    if log:
        histf = [fc]

    for i in range(maxiter):
        try:
            xnext, fnext = linesearch(x, p, f, dfx, f0=fc)
        except ValueError:
            # linsearch failed, resort to fallback method
            logger('%d line-search resort to fallback' % i)
            xnext, fnext = ls_golden(x, p, f, a=0, b=5, f0=fc)
        fc = fnext

        pdfprev = pdfx
        pdfx, dfx = df(xnext)
        if log:
            histf.append(fnext)

        res = np.real(inner(pdfx, pdfx))

        if verbose:
            logger('%4d %16.9f (Ha)  residual: %.3e' % (i, fnext, res))
        x = xnext

        if res < tol:
            logger('minimization: success after', i + 1, ' iterations')
            break

        # conjugate search direction for next iteration
        if restart is not None and i % restart == 0:
            p = -M @ dfx
            assert (np.real(inner(p, dfx)) < 0)
        else:
            b = beta(pdfx, pdfprev, M)
            p = -M @ dfx + b * p
            if np.real(inner(p, dfx)) > 0:
                p = -M @ dfx
                assert np.real(inner(p, dfx)) < 0
    else:
        if log:
            return (x, maxiter, False, histf)
        else:
            return (x, maxiter, False)

    if log:
        return (x, i, True, histf)
    else:
        return (x, i, True)
