kMedoidsParams = {
    'vanilla' : {},
    # removing to improve performances
#     'init=k-medoids++': {'init': 'k-medoids++'},
#     'metric=l1': {'metric': 'l1'},
#     'metric=l2': {'metric': 'l2'},
#     'metric=manhattan': {'metric': 'manhattan'}
}

kmeansParams = {
    'vanilla' : {},
    'tol=5': {'tol': 1e-5},
     # removing to improve performance
#     'n_init30' : {'n_init': 30},
#     'max_iter=400': {'max_iter': 400},
#     'max_iter=600': {'max_iter': 600},
#     'max_iter=400 tol=5': {'max_iter': 400, 'tol': 1e-5},
#     'max_iter=600 tol=5': {'max_iter': 600, 'tol': 1e-5},
#     'max_iter=400 tol=6': {'max_iter': 400, 'tol': 1e-6},
#     'max_iter=600 tol=6': {'max_iter': 600, 'tol': 1e-6},
#     'n_init15 max_iter=400' : {'n_init': 15, 'max_iter': 400},
#     'n_init15 max_iter=600' : {'n_init': 15, 'max_iter': 600},
    
    # removing because of low/similar results
#     'tol=6': {'tol': 1e-6},
#     'n_init15' : {'n_init': 15},
#     'n_init60' : {'n_init': 60},
#     'max_iter=800': {'max_iter': 800},
#     'max_iter=800 tol=5': {'max_iter': 800, 'tol': 1e-5},
#     'max_iter=800 tol=6': {'max_iter': 800, 'tol': 1e-6},
#     'n_init15 max_iter=800' : {'n_init': 15, 'max_iter': 800},
#     'n_init30 max_iter=400' : {'n_init': 30, 'max_iter': 400},
#     'n_init30 max_iter=600' : {'n_init': 30, 'max_iter': 600},
#     'n_init30 max_iter=800' : {'n_init': 30, 'max_iter': 800},
#     'n_init60 max_iter=400' : {'n_init': 60, 'max_iter': 400},
#     'n_init60 max_iter=600' : {'n_init': 60, 'max_iter': 600},
#     'n_init60 max_iter=800' : {'n_init': 60, 'max_iter': 800},
}

spectralClusteringParams = {
    'affinity=laplacian': {'affinity': 'laplacian'},
    # removing to improve performance
    # 'vanilla' : {},
    # 'eigen_solver=arpack': {'eigen_solver': 'arpack'},
    # 'eigen_solver=amg': {'eigen_solver': 'amg'},
#     'assign_labels=discretize': {'assign_labels': 'discretize'},
#     'eigen_solver=lobpcg': {'eigen_solver': 'lobpcg'}, # not working for k>=6
}

agglomerativeClusteringParams = {
    'vanilla' : {},
    
    # removing to improve performance
    'linkage=average affinity=l1' : {'linkage': 'average', 'affinity': 'l1'},
    'linkage=average affinity=manhattan' : {'linkage': 'average', 'affinity': 'manhattan'},
    
    # removing because of low results
    'linkage=complete' : {'linkage': 'complete'},
    'linkage=average' : {'linkage': 'average'},
    'linkage=single' : {'linkage': 'single'},
    'linkage=complete affinity=l1' : {'linkage': 'complete', 'affinity': 'l1'},
    'linkage=single affinity=l1' : {'linkage': 'single', 'affinity': 'l1'},
    'linkage=complete affinity=l2' : {'linkage': 'complete', 'affinity': 'l2'},
    'linkage=average affinity=l2' : {'linkage': 'average', 'affinity': 'l2'},
    'linkage=single affinity=l2' : {'linkage': 'single', 'affinity': 'l2'},
    'linkage=complete affinity=manhattan' : {'linkage': 'complete', 'affinity': 'manhattan'},
    'linkage=single affinity=manhattan' : {'linkage': 'single', 'affinity': 'manhattan'},
    'linkage=complete affinity=cosine' : {'linkage': 'complete', 'affinity': 'cosine'},
    'linkage=average affinity=cosine' : {'linkage': 'average', 'affinity': 'cosine'},
    'linkage=single affinity=cosine' : {'linkage': 'single', 'affinity': 'cosine'},
}

birchParams = {
    'vanilla': {},
}

dbscanParams = {
    'vanilla': {},
}