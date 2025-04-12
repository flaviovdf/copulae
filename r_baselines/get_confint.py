# -*- coding: utf8 -*-


from scipy.stats import bootstrap


import glob
import numpy as np


baselines = [
    'par',   # VineCopula
    'bern',  # KDE copula
    'T',
    'TLL1',
    'TLL2',
    'TLL2nn',
    'MR',
    'beta',
    'pbern', 'pspl1', 'pspl2'  # penRvine
]


dss = []
for fold in glob.glob('data/*'):
    ds = fold.split('/')[1]
    dss.append(ds)

for ds in sorted(dss):
    print(ds)
    for baseline in baselines:
        copula_density = np.genfromtxt(
            'data/{}/{}_yhat.csv'.format(ds, baseline),
            delimiter=','
        )
        I_pdf = np.genfromtxt(
            'data/{}/marg_tst.csv'.format(ds),
            delimiter=','
        )
        points_density = copula_density * I_pdf
        yhat = -np.log(points_density)

        res = bootstrap((yhat, ), np.mean)
        mean = np.mean(yhat)
        upper = res.confidence_interval[1]
        lower = res.confidence_interval[0]
        ci = 0.5 * (upper - lower)
        cint = f'{mean:.2f} \\pm {ci:.2f}'
        print(baseline, cint)
    print()
