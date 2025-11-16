# train_rf.py — Random Forest + GridSearchCV (solo el modelo ganador)
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

CSV_PATH   = "Planetas_2025-10-21.csv"
MODEL_PATH = "model_final.pkl"
RANDOM_STATE = 42

# ===== 1) Datos =====
df = pd.read_csv(CSV_PATH)
if "id_kepler" in df.columns:
    df = df.set_index("id_kepler")

# Solo clases de interés
df = df[df["disposicion_final"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()

X = df.drop(columns=[c for c in ["disposicion_final","disposicion_pipeline","puntaje_confianza"] if c in df.columns])
y = df["disposicion_final"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ===== 2) Preprocesamiento =====
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        cols = [c for c in self.columns_to_drop if c in X.columns]
        return X.drop(columns=cols, errors="ignore")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

column_transform = ColumnTransformer([
    ("num", numeric_pipeline, make_column_selector(dtype_include=["int64","float64"]))
])

preprocessor = Pipeline([
    ("drop_columns", DropColumns(columns_to_drop=[
        "estado_vetting","fecha_vetting","origen_parametros",
        "origen_disposicion","fecha_procesamiento","edad_estelar_gyr"
    ])),
    ("column_transform", column_transform)
])

# ===== 3) Modelo + GridSearchCV (SOLO Random Forest) =====
pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))
])

# Grilla alineada con tu 3ª entrega (podés ampliarla si querés)
param_grid_rf = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5, 10],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid_rf,
    scoring="f1",          # F1 de la clase positiva (CONFIRMED / FP según orden interno)
    cv=5,
    n_jobs=-1,
    refit=True,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nMejores hiperparámetros (RandomForest):")
print(grid.best_params_)

# ===== 4) Evaluación y guardado =====
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
print("\n=== Classification report (TEST) ===")
print(classification_report(y_test, y_pred, digits=3))
print("=== Matriz de confusión (TEST) ===")
print(confusion_matrix(y_test, y_pred))
print("Clases del modelo:", list(best_model.named_steps["model"].classes_))

joblib.dump(best_model, MODEL_PATH)
print(f"\nModelo guardado en: {MODEL_PATH}")
