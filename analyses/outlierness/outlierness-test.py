#!/usr/bin/python2

'''This is a set of simple experiments to test the outlier-rejection and
sensitivity logic'''


import numpy as np
import numpysane as nps
import gnuplotlib as gp
import sys

reference_equation = '0.1*(x+0.2)**2. + 3.0'

def report_mismatch_relerr(a,b, what):
    relerr = np.abs(a-b) / ( (a+b)/2.)
    if relerr > 1e-6:
        print "MISMATCH for {}: relerr = {}, a = {}, b = {},".format(what,relerr,a,b);

def report_mismatch_abserr(a,b, what):
    abserr = np.abs(a-b)
    if abserr > 1e-6:
        print "MISMATCH for {}: abserr = {}, a = {}, b = {},".format(what,abserr,a,b);

def model_matrix(q, order):
    r'''Returns the model matrix S for particular domain points

    Here the "order" is the number of parameters in the fit. Thus order==2 means
    "linear" and order==3 means "quadratic""

    '''
    q = nps.atleast_dims(q,-1)
    if order == 3:
        return nps.transpose(nps.cat(0*q+1, q, q*q))
    elif order == 2:
        return nps.transpose(nps.cat(0*q+1, q))
    raise Exception("unknown order")

def func_reference(q):
    '''Reference function: reference_equation

    Let's say I care about the range [0..1]. It gets less linear as I get
    further away from 0

    '''

    # needs to manually match 'reference_equation' above
    return 0.1 * (q+0.2)*(q+0.2) + 3.0

def func_hypothesis(q, p):
    '''Hypothesis based on parameters

    '''
    S = model_matrix(q, len(p))
    return nps.matmult(p, nps.transpose(S))

def compute_outliernesses(J, x, jq, k_dima, k_cook):
    '''Computes all the outlierness/Cook's D metrics

    I have 8 things I can compute coming from 3 yes/no choices. These are all
    very similar, with two pairs actually coming out identical. I choose:

    - Are we detecting outliers, or looking at effects of a new query point?
    - Dima's outlierness factor or Cook's D
    - Look ONLY at the effect on the other variables, or on the other variables
      AND self?

    If we're detecting outliers, we REMOVE measurements from the dataset, and
    see what happens to the fit. If we're looking at effects of a new query
    point, we see what happend if we ADD measurements

    Dima's outlierness factor metric looks at what happens to the cost function
    E = norm2(x). Specifically I look at

      (norm2(x_before) - norm(x_after))/Nmeasurements

    Cook's D instead looks at

      (norm2(x_before - x_after)) * k

    for some constant k.

    Finally, we can decide whether to include the effects on the measurements
    we're adding/removing, or not.

    Note that here I only look at adding/removing SCALAR measurements


    =============




    This is similar-to, but not exactly-the-same-as Cook's D. I assume the least
    squares fit optimizes a cost function E = norm2(x). The outlierness factor I
    return is

      f = 1/Nmeasurements (E(outliers and inliers) - E(inliers only))

    For a scalar measurement, this solves to

      k = xo^2 / Nmeasurements
      B = 1.0/(jt inv(JtJ) j - 1)
      f = -k * B

    (see the comment in dogleg_getOutliernessFactors() for a description)

    Note that my metric is proportional to norm2(x_io) - norm2(x_i). This is NOT
    the same as Cook's distance, which is proportional to norm2(x_io - x_i).
    It's not yet obvious to me which is better





    There're several slightly-different definitions of Cook's D and of a
    rule-of-thumb threshold floating around on the internet. Wikipedia says:

      D = norm2(x_io - x_i)^2 / (Nstate * norm2(x_io)/(Nmeasurements - Nstate))
      D_threshold = 1

    An article https://www.nature.com/articles/nmeth.3812 says

      D = norm2(x_io - x_i)^2 / ((Nstate+1) * norm2(x_io)/(Nmeasurements - Nstate -1))
      D_threshold = 4/Nmeasurements

    Here I use the second definition. That definition expands to

      k = xo^2 / ((Nstate+1) * norm2(x_io)/(Nmeasurements - Nstate -1))
      B = 1.0/(jt inv(JtJ) j - 1)
      f = k * (B + B*B)

    '''

    Nmeasurements,Nstate = J.shape

    # The A values for each measurement
    Aoutliers = nps.inner(J, nps.transpose(np.linalg.pinv(J)))
    Aquery    = nps.inner(jq, nps.transpose(np.linalg.solve(nps.matmult(nps.transpose(J),J), nps.transpose(jq))))

    def dima():

        k = k_dima

        def outliers():
            B = 1.0 / (Aoutliers - 1.0)
            return dict( self_others = k * x*x*(-B  ) ,
                         others      = k * x*x*(-B-1))
        def query():
            B = 1.0 / (Aquery + 1.0)
            return dict( self_others = k * (B) ,
                         others      = k * (B-B*B))
        return dict(outliers = outliers(),
                    query    = query())
    def cook():

        k = k_cook

        def outliers():
            B = 1.0 / (Aoutliers - 1.0)
            return dict( self_others = k * x*x*(B+B*B  ) ,
                         others      = k * x*x*(-B-1))
        def query():
            B = 1.0 / (Aquery + 1.0)
            return dict( self_others = k * (1-B) ,
                         others      = k * (B-B*B))
        return dict(outliers = outliers(),
                    query    = query())


    return dict(cook = cook(),
                dima = dima())

def outlierness_test(J, x, f, outlierness, k_dima, k_cook, i=0):
    r'''Test the computation of outlierness

    I have an analytical expression for this computed in
    compute_outliernesses(). This explicitly computes the quantity represented
    by compute_outliernesses() to make sure that that analytical expression is
    correct

    '''

    # I reoptimize without measurement i
    E0 = nps.inner(x,x)

    J1 = nps.glue(J[:i,:], J[(i+1):,:], axis=-2)
    f1 = nps.glue(f[:i  ], f[(i+1):  ], axis=-1)
    p1 = nps.matmult( f1, nps.transpose(np.linalg.pinv(J1)))
    x1 = nps.matmult(p1, nps.transpose(J1)) - f1
    E1 = nps.inner(x1,x1)

    report_mismatch_relerr( (E0-E1) * k_dima,
                            outlierness['self_others'][i],
                            "self_others outlierness computed analytically, explicitly")
    report_mismatch_relerr( (E0-x[i]*x[i] - E1) * k_dima,
                            outlierness['others'][i],
                            "others outlierness computed analytically, explicitly")

def CooksD_test(J, x, f, CooksD, k_dima, k_cook, i=0):
    r'''Test the computation of Cook's D

    I have an analytical expression for this computed in
    compute_outliernesses(). This explicitly computes the quantity represented
    by compute_outliernesses() to make sure that that analytical expression is
    correct

    '''

    # I reoptimize without measurement i
    Nmeasurements,Nstate = J.shape

    J1 = nps.glue(J[:i,:], J[(i+1):,:], axis=-2)
    f1 = nps.glue(f[:i  ], f[(i+1):  ], axis=-1)
    p1 = nps.matmult( f1, nps.transpose(np.linalg.pinv(J1)))
    x1 = nps.matmult(p1, nps.transpose(J)) - f

    dx = x1-x

    report_mismatch_relerr( nps.inner(dx,dx) * k_cook,
                            CooksD['self_others'][i],
                            "self_others CooksD computed analytically, explicitly")
    report_mismatch_relerr( (nps.inner(dx,dx) - dx[i]*dx[i]) * k_cook,
                            CooksD['others'][i],
                            "others CooksD computed analytically, explicitly")

def outlierness_query_test(J,p,x, f, query,fquery_ref, outlierness_nox, k_dima, k_cook, i=0):
    r'''Test the concept of outlierness for querying hypothetical data

    fquery_test = f(q) isn't true here. If it WERE true, the x of the query
    point would be 0 (we fit the model exactly), so the outlierness factor would
    be 0 also

    '''

    # current solve
    E0 = nps.inner(x,x)

    query      = query     [i]
    fquery_ref = fquery_ref[i]


    # I add a new point, and reoptimize
    fquery = func_hypothesis(query,p)
    xquery = fquery - fquery_ref
    jquery = model_matrix(query, len(p))

    J1 = nps.glue(J, jquery,     axis=-2)
    f1 = nps.glue(f, fquery_ref, axis=-1)
    p1 = nps.matmult( f1, nps.transpose(np.linalg.pinv(J1)))
    x1 = nps.matmult(p1, nps.transpose(J1)) - f1
    E1 = nps.inner(x1,x1)

    report_mismatch_relerr( (E1 - E0) * k_dima,
                            outlierness_nox['self_others'][i]*xquery*xquery,
                            "self_others query-outlierness computed analytically, explicitly")
    report_mismatch_relerr( (E1-x1[-1]*x1[-1] - E0) * k_dima,
                            outlierness_nox['others'][i]*xquery*xquery,
                            "others query-outlierness computed analytically, explicitly")

def CooksD_query_test(J,p,x, f, query,fquery_ref, CooksD_nox, k_dima, k_cook, i=0):
    r'''Test the concept of CooksD for querying hypothetical data

    fquery_test = f(q) isn't true here. If it WERE true, the x of the query
    point would be 0 (we fit the model exactly), so the outlierness factor would
    be 0 also

    '''

    # current solve
    Nmeasurements,Nstate = J.shape

    query      = query     [i]
    fquery_ref = fquery_ref[i]

    # I add a new point, and reoptimize
    fquery = func_hypothesis(query,p)
    xquery = fquery - fquery_ref
    jquery = model_matrix(query, len(p))

    J1 = nps.glue(J, jquery,     axis=-2)
    f1 = nps.glue(f, fquery_ref, axis=-1)
    p1 = nps.matmult( f1, nps.transpose(np.linalg.pinv(J1)))
    x1 = nps.matmult(p1, nps.transpose(J1)) - f1

    dx = x1[:-1] - x

    dx_both = x1 - nps.glue(x,xquery, axis=-1)

    report_mismatch_relerr( nps.inner(dx_both,dx_both)*k_cook,
                            CooksD_nox['self_others'][i]*xquery*xquery,
                            "self_others query-CooksD computed analytically, explicitly")
    report_mismatch_relerr( nps.inner(dx,dx)*k_cook,
                            CooksD_nox['others'][i]*xquery*xquery,
                            "others query-CooksD computed analytically, explicitly")



def Var_df(J, squery, stdev):
    r'''Propagates noise in input to noise in f

    noise in input -> noise in params -> noise in f

    dp ~ M dm where M = inv(JtJ)Jt

    df = df/dp dp

    df/dp = squery
    Var(dm) = stdev^2 I ->

    Var(df) = stdev^2 squery inv(JtJ) Jt J inv(JtJ) squeryt =
            = stdev^2 squery inv(JtJ) squeryt

    This function broadcasts over squery
    '''
    return \
        nps.inner(squery,
                  nps.transpose(np.linalg.solve(nps.matmult(nps.transpose(J),J),
                                                nps.transpose(squery)))) *stdev*stdev

def histogram(x, **kwargs):
    h,edges = np.histogram(x, bins=20, **kwargs)
    centers = (edges[1:] + edges[0:-1])/2
    return h,centers,edges[1]-edges[0]


def generate_dataset(N, noise_stdev):
    q      = np.random.rand(N)
    fref   = func_reference(q)
    fnoise = np.random.randn(N) * noise_stdev
    f      = fref + fnoise
    return q,f


def fit(q, f, order):
    S = model_matrix(q, order)
    J = S
    p = nps.matmult( f, nps.transpose(np.linalg.pinv(S)))
    x = func_hypothesis(q,p) - f
    return p,J,x




N           = 100   # This many points in the dataset
noise_stdev = 0.001 # The dataset is corrupted thusly
Nquery      = 70    # This many points for post-fit uncertainty-evaluation


def test_order(q,f, query, order):

    # I look at linear and quadratic models: a0 + a1 q + a2 q^2, with a2=0 for the
    # linear case. I use plain least squares. The parameter vector is [a0 a1 a2]t. S
    # = [1 q q^2], so the meaurement vector x = S p - f. E = norm2(x). J = dx/dp =
    # S.
    #
    # Note the problem "order" is the number of parameters, so a linear model has
    # order==2
    p,J,x = fit(q,f,order)

    Nmeasurements,Nstate = J.shape
    k_dima = 1.0/Nmeasurements
    k_cook = 1.0/((Nstate + 1.0) * nps.inner(x,x)/(Nmeasurements - Nstate - 1.0))

    report_mismatch_abserr(np.linalg.norm(nps.matmult(x,J)), 0, "Jtx")

    squery = model_matrix(query, order)
    fquery = func_hypothesis(query, p)
    metrics = compute_outliernesses(J,x, squery, k_dima, k_cook)

    outlierness_test(J, x, f, metrics['dima']['outliers'], k_dima, k_cook, i=10)
    CooksD_test     (J, x, f, metrics['cook']['outliers'], k_dima, k_cook, i=10)
    outlierness_query_test(J,p,x,f, query, fquery + 1.2e-3, metrics['dima']['query'], k_dima, k_cook, i=10 )
    CooksD_query_test     (J,p,x,f, query, fquery + 1.2e-3, metrics['cook']['query'], k_dima, k_cook, i=10 )

    Vquery = Var_df(J, squery, noise_stdev)
    return p,J,x,Vquery,squery,fquery,metrics,k_dima,k_cook



q,f   = generate_dataset(N, noise_stdev)
query = np.linspace(-1,2, Nquery)

p2,J2,x2,Vquery2,squery2,fquery2,metrics2,k_dima2,k_cook2 = test_order(q,f, query, 2)
p3,J3,x3,Vquery3,squery3,fquery3,metrics3,k_dima3,k_cook3 = test_order(q,f, query, 3)


def show_outlierness(q, metrics, Nmeasurements, dimas_threshold, cooks_threshold):

    gp.plot( (q, metrics['dima']['outliers']['self_others']/dimas_threshold,
              dict(legend="Dima's self+others outlierness / threshold",
                   _with='points')),
             (q, metrics['dima']['outliers']['others']/dimas_threshold,
              dict(legend="Dima's others-ONLY outlierness / threshold",
                   _with='points')),
             (q, metrics['cook']['outliers']['self_others']/cooks_threshold,
              dict(legend="Cook's self+others outlierness / threshold",
                   _with='points')),
             (q, metrics['cook']['outliers']['others']/cooks_threshold,
              dict(legend="Cook's others-ONLY outlierness / threshold",
                   _with='points')),

             equation='1.0 title "Threshold"',
             wait = 1)


def show_uncertainty(Vquery, metrics, Nmeasurements, dimas_threshold, cooks_threshold):
    gp.plot(

             (query, np.sqrt(Vquery),
              dict(legend='expectederr (y2)', _with='lines', y2=1)),

             (query, metrics['dima']['query']['self_others']*noise_stdev*noise_stdev / dimas_threshold,
              dict(legend="Dima's self+others query / threshold",
                   _with='linespoints')),
             (query, metrics['dima']['query']['others']*noise_stdev*noise_stdev / dimas_threshold,
              dict(legend="Dima's others-ONLY query / threshold",
                   _with='linespoints')),

             (query, metrics['cook']['query']['self_others']*noise_stdev*noise_stdev / cooks_threshold,
              dict(legend="Cook's self+others query / threshold",
                   _with='linespoints')),
             (query, metrics['cook']['query']['others']*noise_stdev*noise_stdev / cooks_threshold,
              dict(legend="Cook's others-ONLY query / threshold",
                   _with='linespoints')),

             equation='1 title "Threshold"',
             wait = 1)

def show_fit(q, fdata, query, fqueryfit, Vquery):
    gp.plot((q,fdata, dict(_with='points', legend = 'input data')),
            (query, fqueryfit + np.sqrt(Vquery)*np.array(((1,),(0,),(-1,),)), dict(legend = 'stdev_f', _with='lines')),
            equation='{} with lines title "reference"'.format(reference_equation),
            xrange=[-1,2],
            wait = 1)

def show_distribution(outlierness):
    h,c,w = histogram(outlierness)
    raise Exception("not done")
    #gp.plot()



# This is hoaky, but reasonable, I think. Some of the unscaled metrics are
# identical between mine and Cook's expressions. So I scale Cook's 4/N threshold
# to apply to me. Works ok.
cooks_threshold  = 4.0 / N
dimas_threshold2 = cooks_threshold / k_cook2 * k_dima2
dimas_threshold3 = cooks_threshold / k_cook3 * k_dima3

# These all show interesting things; turn one of them on
show_outlierness(q, metrics2, N, dimas_threshold2, cooks_threshold)
# show_fit(q,f,query,fquery2,Vquery2)
# show_uncertainty(Vquery3, metrics3, N, dimas_threshold3, cooks_threshold)


# Conclusions:
#
# - Cook's 4/N threshold looks reasonable.
#
# - For detecting outliers my self+others metric is way too outlier-happy. It
#   makes most of the linear fit points look outliery
#
# - For uncertainty, Dima's self+others is the only query metric that's maybe
#   usable. It qualitatively a negated and shifted Cook's self+others
