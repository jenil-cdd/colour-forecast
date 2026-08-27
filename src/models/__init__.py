"""Model registry.

Every forecaster is registered here so the backtest can run the whole suite by
name and produce a single comparable leaderboard.
"""

from __future__ import annotations

from src.models.base import Forecaster, Prediction
from src.models.baselines import (
    ElasticNetVelocity,
    HeuristicOrderModel,
    LassoVelocity,
    NaiveFamilyMean,
    RidgeVelocity,
    SizeRatioBaseline,
)
from src.models.bayes import HierarchicalBayesVelocity
from src.models.cluster import ClusterPoolMean
from src.models.counts import NegativeBinomialGLM, PoissonGLM
from src.models.growth import BassDiffusion, GompertzRamp, LogisticRamp
from src.models.lookalike import KNNLookalike
from src.models.mf import MatrixFactorisation
from src.models.trees import LightGBMForecaster, RandomForestForecaster, XGBoostForecaster
from src.models.ensemble import StackedEnsemble

REGISTRY: dict[str, type[Forecaster]] = {
    # --- benchmarks -------------------------------------------------------
    "heuristic": HeuristicOrderModel,
    "naive_family": NaiveFamilyMean,
    "size_ratio": SizeRatioBaseline,
    # --- regularised linear ----------------------------------------------
    "ridge": RidgeVelocity,
    "lasso": LassoVelocity,
    "elasticnet": ElasticNetVelocity,
    # --- count GLMs -------------------------------------------------------
    "poisson_glm": PoissonGLM,
    "negbin_glm": NegativeBinomialGLM,
    # --- trees ------------------------------------------------------------
    "random_forest": RandomForestForecaster,
    "xgboost": XGBoostForecaster,
    "lightgbm": LightGBMForecaster,
    # --- growth / cold-start curves ---------------------------------------
    "gompertz": GompertzRamp,
    "logistic": LogisticRamp,
    "bass": BassDiffusion,
    # --- uncertainty / borrowing strength ---------------------------------
    "hier_bayes": HierarchicalBayesVelocity,
    # --- instance based ---------------------------------------------------
    "knn_lookalike": KNNLookalike,
    # --- latent structure -------------------------------------------------
    "matrix_factorisation": MatrixFactorisation,
    "cluster_pool": ClusterPoolMean,
    # --- combination ------------------------------------------------------
    "ensemble": StackedEnsemble,
}

__all__ = ["REGISTRY", "Forecaster", "Prediction"]
