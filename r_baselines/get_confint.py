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


for fold in glob.glob('data/*'):
    ds = fold.split('/')[1]
    for baseline in baselines:
        copula_density = np.genfromtxt(
            'data/{}/{}_yhat.csv'.format(ds, baseline),
            delimiter=','
        )

        I_pdf = np.genfromtxt(
            'data/{}/marg_tst.csv'.format(ds)
        )
        points_density = copula_density * I_pdf
        yhat = -np.log(points_density)

        res = bootstrap(yhat, np.nanmean)
        print(baseline)
        print(
            np.nanmean(yhat),
            res.confidence_interval[0],
            res.confidence_interval[1]
        )
