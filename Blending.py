import numpy as np
from sklearn.linear_model import LogisticRegression

class BlendingClassifier:
    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else LogisticRegression()
    
    def fit(self, X, y):
        # Split simples (holdout) para gerar meta-features
        from sklearn.model_selection import train_test_split
        X_train, X_blend, y_train, y_blend = train_test_split(X, y, test_size=0.3, random_state=42)

        # Treinar os modelos de base
        self.fitted_base = []
        blend_features = []
        for name, model in self.base_models:
            m = model.fit(X_train, y_train)
            self.fitted_base.append((name, m))
            preds = m.predict_proba(X_blend)
            blend_features.append(preds)
        
        # Criar features para meta-modelo
        blend_features = np.hstack(blend_features)

        # Treinar meta-modelo
        self.meta_model.fit(blend_features, y_blend)
        return self
    
    def predict(self, X):
        blend_features = []
        for _, m in self.fitted_base:
            preds = m.predict_proba(X)
            blend_features.append(preds)
        blend_features = np.hstack(blend_features)
        return self.meta_model.predict(blend_features)
    
    def predict_proba(self, X):
        blend_features = []
        for _, m in self.fitted_base:
            preds = m.predict_proba(X)
            blend_features.append(preds)
        blend_features = np.hstack(blend_features)
        return self.meta_model.predict_proba(blend_features)
