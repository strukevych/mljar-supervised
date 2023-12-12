import logging

import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.svm import SVC, SVR

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
    AlgorithmsRegistry,
)
from supervised.algorithms.sklearn import SklearnAlgorithm
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

class SVMFit(SklearnAlgorithm):
    def file_extension(self):
        return "svm"

    def is_fitted(self):
        return (
            hasattr(self.model, "fit_status_")
            and self.model.fit_status_ is not None
            and self.model.fit_statis_ == 0
        )

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_validation=None,
        y_validation=None,
        sample_weight_validation=None,
        log_to_file=None,
        max_time=None,
    ):
        self.model.fit(X, y, sample_weight)


class SVMAlgorithm(SVMFit, ClassifierMixin):
    algorithm_name = "Support Vector Machine"
    algorithm_short_name = "SVM"

    def __init__(self, params):
        super(SVMAlgorithm, self).__init__(params)
        logger.debug("SVMAlgorithm.__init__")
        self.library_version = sklearn.__version__
        self.model = SVC(
            C=params.get("C", 1.0),
            coef0=params.get("coef0", 0.0),
            degree=params.get("degree", 3),
            gamma=params.get("gamma", "scale"),
            tol=params.get("tol", 1e-3),
            kernel=params.get("kernel", "rbf"),
            probability=params.get("probability", False),
            random_state=1234,
            n_jobs=params.get("n_jobs", -1),
        )


class SVMRegressorAlgorithm(SVMFit, RegressorMixin):
    algorithm_name = "Support Vector Machine"
    algorithm_short_name = "SVM"

    def __init__(self, params):
        super(SVMRegressorAlgorithm, self).__init__(params)
        logger.debug("SVMRegressorAlgorithm.__init__")
        self.library_version = sklearn.__version__
        self.model = SVR(
            C=params.get("C", 1.0),
            coef0=params.get("coef0", 0.0),
            degree=params.get("degree", 3),
            gamma=params.get("gamma", "scale"),
            tol=params.get("tol", 1e-3),
            kernel=params.get("kernel", "rbf"),
            epsilon=params.get("epsilon", 0.1),
            random_state=1234,
            n_jobs=params.get("n_jobs", -1),
        )


svm_params = { "C": [0.1, 1, 10, 100], "coef0": [0.0], "degree": [3], "gamma": ["scale", "auto"], "tol": [1e-3, 1e-4, 1e-5], "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"] }
svm_params_regression = svm_params.copy()
svm_params_regression["epsilon"] = [0.1, 0.2, 0.5]
svm_params_multiclass = svm_params.copy()
svm_params_multiclass["probability"] = [True]

default_params = { "C": 1.0, "coef0": 0.0, "kernel": "rbf", "gamma": "scale", "tol": 1e-3, "degree": 3 }
default_params_multiclass = default_params.copy()
default_params_multiclass = { "probability": True }
default_params_regression = default_params.copy()
default_params_regression["epsilon"] = 0.1

additional = {"max_rows_limit": None, "max_cols_limit": None}

required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "scale",
    "target_as_integer",
]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    SVMAlgorithm,
    svm_params,
    required_preprocessing,
    additional,
    default_params,
)
AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    SVMAlgorithm,
    svm_params_multiclass,
    required_preprocessing,
    additional,
    default_params_multiclass,
)

AlgorithmsRegistry.add(
    REGRESSION,
    SVMRegressorAlgorithm,
    svm_params_regression,
    required_preprocessing,
    additional,
    svm_params_regression,
)
