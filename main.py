import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor
)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true))==2 else (0,0,0,0)
    espec = tn/(tn+fp) if (tn+fp)>0 else np.nan
    print(f"tn = {tn} \n fp = {fp} \n")
    #Imprimir matriz de confusão para testar
    confusion = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(confusion)


    return acc, f1, prec, rec, espec

def regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mse, rmse, mae

# ============================================================
# CLASSIFICAÇÃO (Hold-out e CrossVal)
# ============================================================

# Função de especificidade
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):  # só faz sentido para classificação binária
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn+fp) > 0 else 0
    else:
        # para multi-classe: calcula média da especificidade "classe vs resto"
        especs = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
            fp = cm[:,i].sum() - cm[i,i]
            especs.append(tn / (tn + fp) if (tn+fp) > 0 else 0)
        return np.mean(especs)

specificity = make_scorer(specificity_score)

def run_classification(X, y, cv=False):
    results = []
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "MLP": MLPClassifier(max_iter=1000),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Bagging": BaggingClassifier(),
        "Boosting": AdaBoostClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
    }

    # Ensemble simples (Voting)
    ens = VotingClassifier(estimators=[
        ('knn', KNeighborsClassifier()),
        ('dt', DecisionTreeClassifier()),
        ('nb', GaussianNB())
    ], voting='hard')
    models["Ensemble"] = ens

    # Stacking
    models["Stacking"] = StackingClassifier(
        estimators=[('knn', KNeighborsClassifier()), ('dt', DecisionTreeClassifier()), ('nb', GaussianNB())],
        final_estimator=LogisticRegression()
    )

    # Blending (simulado via Voting sonft)
    models["Blending"] = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier()),
        ('mlp', MLPClassifier(max_iter=500))
    ], voting='soft')

    if cv:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'specificity': specificity
        }
        for name, model in models.items():
            scores = cross_validate(model, X, y, cv=kf, scoring=scoring)
            results.append([name,
                            scores['test_accuracy'].mean(),
                            scores['test_f1_weighted'].mean(),
                            scores['test_precision_weighted'].mean(),
                            scores['test_recall_weighted'].mean(),
                            scores['test_specificity'].mean()])

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc, f1, prec, rec, espec = classification_metrics(y_test, y_pred)
            results.append([name, acc, f1, prec, rec, espec])

    df = pd.DataFrame(results, columns=["Modelo","Acurácia","F1","Precisão","Sensibilidade","Especificidade"])
    return df

# ============================================================
# REGRESSÃO (Hold-out e CrossVal)
# ============================================================

def run_regression(X, y, cv=False):
    results = []
    models = {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Ridge": Ridge(),
        "MLP": MLPRegressor(max_iter=1000),
        "SVM": SVR(),
        "Random Forest": RandomForestRegressor(),
        "Bagging": BaggingRegressor(),
        "Boosting": AdaBoostRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
    }

    # Ensemble simples (média de regressões)
    models["Ensemble"] = VotingRegressor([
        ('lr', LinearRegression()),
        ('dt', DecisionTreeRegressor())
    ])

    # Stacking
    models["Stacking"] = StackingRegressor(
        estimators=[('knn', KNeighborsRegressor()), ('dt', DecisionTreeRegressor())],
        final_estimator=LinearRegression()
    )

    # Blending (simulado com Voting)
    models["Blending"] = VotingRegressor([
        ('rf', RandomForestRegressor()),
        ('mlp', MLPRegressor(max_iter=500))
    ])

    if cv:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in models.items():
            scores = cross_validate(model, X, y, cv=kf,
                                    scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'])
            results.append([name,
                            scores['test_r2'].mean(),
                            -scores['test_neg_mean_squared_error'].mean(),
                            np.sqrt(-scores['test_neg_mean_squared_error'].mean()),
                            -scores['test_neg_mean_absolute_error'].mean()])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2, mse, rmse, mae = regression_metrics(y_test, y_pred)
            results.append([name, r2, mse, rmse, mae])

    df = pd.DataFrame(results, columns=["Modelo","R2","MSE","RMSE","MAE"])
    return df

# ============================================================
# EXEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    # ---------- CLASSIFICAÇÃO ----------
    data = pd.read_csv("Datasets/student-mat.csv", sep=";")
    data["pass"] = (data["G3"] >= 10).astype(int)
    X = data.drop(columns=["pass","G3"])
    X = pd.get_dummies(X, drop_first=True)
    y = data["pass"]

    df_class_holdout = run_classification(X, y, cv=False)
    df_class_cv = run_classification(X, y, cv=True)

    df_class_holdout.to_csv("results/classificacao_holdout.csv", index=False)
    df_class_cv.to_csv("results/classificacao_crossval.csv", index=False)

    # ---------- REGRESSÃO ----------
    data = pd.read_csv("Datasets/winequality-red.csv", sep=";")
    X = data.drop(columns=["quality"])
    y = data["quality"]

    df_reg_holdout = run_regression(X, y, cv=False)
    df_reg_cv = run_regression(X, y, cv=True)

    df_reg_holdout.to_csv("results/regressao_holdout.csv", index=False)
    df_reg_cv.to_csv("results/regressao_crossval.csv", index=False)

    print("✅ Tabelas exportadas como CSV!")
