
EXEC sp_execute_external_script
    @language = N'Python',
    @script = N'df=InputDataSet
OutputDataSet=InputDataSet
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Tiền xử lý dữ liệu
tfidf = TfidfVectorizer(stop_words="english", min_df=2)
X = tfidf.fit_transform(df["text"].values.astype("U"))
y = df["label"]

# Phân chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Xây dựng mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")

# Đào tạo mô hình KNN
knn.fit(X_train, y_train)

# Đánh giá mô hình KNN
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của mô hình KNN:", accuracy)
',
@input_data_1=N'select top (1000) * from dbo.FakeNews'
with result sets((
[column1] int
,[title] nvarchar(50)
,[text] NVARCHAR(50)
,[label] int))