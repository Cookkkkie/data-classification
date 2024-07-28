import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib


def predict_species(new_data):
    return best_model.predict(new_data)


df = pd.read_csv('penguins_size.csv')


df = df.dropna()


df['sex'] = df['sex'].map({'MALE': 0, 'FEMALE': 1})
df['island'] = df['island'].map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})


sns.pairplot(df, hue='species')
plt.show()


X = df.drop(columns=['species'])
y = df['species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Начальная модель:")
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8]
}


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_


y_pred_best = best_model.predict(X_test)
print("Модель после подбора гиперпараметров:")
print(accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))


joblib.dump(best_model, 'penguin_classifier_model.pkl')
