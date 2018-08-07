import numpy as np


def inner(a, b):
    """
    complex inner product
    """
    return np.einsum('ij,ij->', a, np.conj(b))


def beta_fletcher_reves(dfnext, df):
    """
    """
    return np.asscalar(inner(dfnext, dfnext) / inner(df, df))


def beta_polak_ribiere(dfnext, df):
    """
    """
    return np.asscalar(inner(dfnr, dfnr-dfr) / inner(df, df))


def beta_sd(dfnext, df):
    """
    """
    return 0


def ls_qinterp(x0, p, f, dfx0, s=0.2):
    """
    Keyword Arguments:
    x0       -- starting point
    p        -- search direction
    f        -- f(x)
    dfx0     -- grad f(x0)
    """
    f0 = f(x0)
    # TODO: remove hard-coded band occ numbers!
    b = np.real(2 * inner(p, dfx0))
    assert (b <= 0)
    f1 = f(x0 + s * p)
    # fit coefficients g(t) = a * t^2 + b * t + c
    c = f0
    a = (f1 - b * s - c) / s**2
    # min g(t)
    tmin = -b / (2 * a)

    if a <= 0:
        raise ValueError('quadratic interpolation: not convex!')

    x = x0 + tmin * p
    print('ls_qinterp::tmin:', tmin)
    fnext = f(x)
    if not fnext < f0:
        print('ls_qinterp failed!!!')
        # revert coefficients to x0
        f0r = f(x0)  # never remove this, need side effects
        assert (f0 == f0r)

        raise ValueError('qinterp did not improve the solution')
    return x


def ls_golden(x0, dfx0, f):
    """
    Golden-section search
    """
    raise Exception('not yet implemented')


def ls_bracketing(x0, p, f, dfx0, tau=0.5, maxiter=40):
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
    ds = 1024
    print('bracketing: ds initial = ', ds)
    assert (maxiter >= 1)

    fn = f(x0 + ds * p)
    for i in range(maxiter):
        if fn <= f0 + c * ds * m:
            break
        else:
            ds *= tau
            fn = f(x0 + ds * p)
            print('ls_bracketing: ds: %.4g' % ds)
    else:
        if not fn < f0 and np.abs(fn - f0) > 1e-10:
            raise ValueError(
                'failed to find a step length after maxiter=%d iterations.' %
                maxiter)
    print('bracketing: step-length = : ', ds)
    return x0 + ds * p


def minimize(x0,
             f,
             df,
             maxiter=100,
             tol=1e-7,
             lstype='interp',
             mtype='FR',
             restart=20,
             callback=None,
             verbose=False,
             log=False):
    """
    Keyword Arguments:
    x0         -- np.ndarray
    f          -- function object
    df         -- function object
    maxiter    -- (default 100)
    tol        -- (default 1e-7)
    lstype     -- (default 'interp')
    mtype      -- (default 'FR')
    callback   -- (default None)
    verbose    -- (default False) debug output
    log        -- (default False) log values

    Returns: (x, iter, success)
    x          -- result
    iter       -- iteration count
    success    -- converged to specified tolerance
    [history]  -- if log==True
    """

    if mtype == 'FR':
        beta = beta_fletcher_reves
    elif mtype == 'PR':
        beta = beta_polak_ribiere
    elif mtype == 'SD':
        beta = beta_sd
    else:
        raise Exception('invalid argument')

    if lstype == 'interp':
        linesearch = ls_qinterp
    elif lstype == 'golden':
        linesearch = ls_golden
    elif lstype == 'bracketing':
        linesearch = ls_bracketing
    else:
        raise Exception('invalid argument')

    x = x0
    pdfx, dfx = df(x)
    p = -1 * pdfx

    if log:
        histf = [f(x)]

    for i in range(maxiter):
        print('---- minimize iteration ', i)
        try:
            xnext = linesearch(x, p, f, dfx)
        except ValueError:
            # fall back to bracketing line-search
            print('ls_qinterp failed')
            assert (linesearch is not ls_bracketing)
            xnext = ls_bracketing(x, p, f, dfx)

        # TODO update k-point set, e.g. call f(x),
        pdfx, dfx = df(xnext)
        # side effect (update coefficients, band energies, density, potential)
        fnext = f(xnext)
        if log:
            histf.append(fnext)
        if verbose:
            print('current energy: ', i, fnext)
        x = xnext

        res = inner(pdfx, pdfx)
        print('residual: %.4g' % res)
        if res < tol:
            print('success after', i + 1, ' iterations')
            break

        # conjugate search direction for next iteration
        b = beta(-pdfx, p)
        if i % restart == 0:
            p = -pdfx
        else:
            p = -pdfx + b * p

        if callback is not None:
            callback(x)
    else:
        if log:
            return (x, i, False, histf)
        else:
            return (x, i, False)

    if log:
        return (x, i, True, histf)
    else:
        return (x, i, True)
