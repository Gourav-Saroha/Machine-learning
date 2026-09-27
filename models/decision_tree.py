from __future__ import annotations

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text


TREE_PARAM_GRID = {
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_leaf": [1, 2, 4, 6],
}


def build_decision_tree(
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


def tune_decision_tree(
    X_train,
    y_train,
    random_state: int = 42,
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=build_decision_tree(random_state=random_state),
        param_grid=TREE_PARAM_GRID,
        cv=5,
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)
    return grid


def export_rules(model: DecisionTreeClassifier, feature_names: list[str], max_depth: int = 4) -> str:
    return export_text(model, feature_names=feature_names, max_depth=max_depth)
