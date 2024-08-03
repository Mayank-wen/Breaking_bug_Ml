import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    features = df.drop('target', axis=1)
    target = df['target']
    
    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    return preprocessor, features, target

def train_model(preprocessor, features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])
    
    param_grid = {
        'classifier__n_estimators': [10, 50, 100],
        'classifier__max_depth': [None, 10, 20, 30]
    }
    
    search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    
    print(f"Best parameters: {search.best_params_}")
    
    y_pred = search.predict(X_test)
    print(classification_report(y_test, y_pred))

def main():
    df = load_data('data.csv')
    preprocessor, features, target = preprocess_data(df)
    train_model(preprocessor, features, target)

if __name__ == "__main__":
    main()
