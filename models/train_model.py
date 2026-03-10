from sklearn.svm import SVC
def train_svm(x_train,y_train,c,gamma,kernal):
    model = SVC(kernel='rbf',C=10,gamma='scale',random_state=42)
    model.fit(x_train,y_train)

    return model

'''
svm = SVC(kernel='rbf',C=10,gamma='scale',random_state=42)
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report
accuracy = accuracy_score(y_test,y_pred)
print(f'accuracy : {accuracy * 100:.2f}%')
print(classification_report(y_test,y_pred)) '''