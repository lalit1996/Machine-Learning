import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

model = joblib.load('cancer_detection.pkl')


x_new = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0]])

# print(x_new)

y_pred = model.predict(x_new)
print(model.predict_proba(x_new))
print(y_pred)