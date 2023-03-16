
EXEC sp_execute_external_script
    @language = N'Python',
    @script = N'df=InputDataSet
OutputDataSet=InputDataSet
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(df["text"].values.astype("U"))
y = df["label"]

# Phân chia du lieu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Xây dung mô hình SVM
svm = LinearSVC()

# Ðào tao mô hình SVM
svm.fit(X_train, y_train)

# Ðánh giá mô hình SVM
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Ð? chính xác của mô hình SVM: ", accuracy)

',
@input_data_1=N'select top (2000) * from dbo.FakeNews'
with result sets((
[column1] int
,[title] nvarchar(50)
,[text] NVARCHAR(50)
,[label] int))