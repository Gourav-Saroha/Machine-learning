from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 10],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [1, 2, 4],
}


def build_random_forest(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | int | float = "sqrt",
    random_state: int = 42,
    oob_score: bool = True,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        oob_score=oob_score,
        random_state=random_state,
    )


def tune_random_forest(
    X_train,
    y_train,
    random_state: int = 42,
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=build_random_forest(random_state=random_state),
        param_grid=RANDOM_FOREST_PARAM_GRID,
        cv=5,
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)
    return grid
