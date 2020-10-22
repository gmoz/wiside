import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import time
import os
import logging


def setup_logger(name, log_file, formatter, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


fmt_info = '[%(asctime)s]: %(message)s'
fmt_error = '[%(asctime)s] - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
logger_info = setup_logger(__name__ + '_info', 'modeling.info', logging.Formatter(fmt_info))
logger_error = setup_logger(__name__ + '_error', 'modeling.error', logging.Formatter(fmt_error), level=logging.WARNING)

if __name__ == "__main__":
    print(f"Start, current process id is {os.getpid()}.")
    t0 = time.time()

    from inference import Polynomial_feature, build_pipeline, build_grid_search, run_grid_search, pretty_cv_results, \
        test_performance

    print("LinearRegression")
    logger_info.info("LinearRegression")

    pipeline = build_pipeline()
    param_grid = [{
        'Polynomial_feature__degree': [1, 2],
        'Polynomial_feature__interaction_only': [True, False],
        'Polynomial_feature__number_feature': [1, 2, 4],
        'Scaler': [MinMaxScaler(), None],
        'model': [LinearRegression()],
        'model__fit_intercept': [True, False],
        'model__normalize': [True, False],
    }]
    lm_grid_search = build_grid_search(pipeline, param_grid, n_jobs=4)
    run_grid_search(lm_grid_search)

    test_performance(lm_grid_search)
    print(f"Done all in {time.time() - t0:.3f} sec.")
