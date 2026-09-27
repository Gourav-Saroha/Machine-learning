from __future__ import annotations

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


BAGGING_PARAM_GRID = {
    "n_estimators": [10, 50, 100],
    "max_samples": [0.7, 1.0],
    "max_features": [0.7, 1.0],
}


def build_bagging(
    n_estimators: int = 100,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    random_state: int = 42,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    oob_score: bool = True,
) -> BaggingClassifier:
    base_tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    return BaggingClassifier(
        estimator=base_tree,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=True,
        oob_score=oob_score,
        random_state=random_state,
    )


def tune_bagging(
    X_train,
    y_train,
    random_state: int = 42,
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=build_bagging(random_state=random_state),
        param_grid=BAGGING_PARAM_GRID,
        cv=5,
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)
    return grid
