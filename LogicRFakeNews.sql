EXEC sp_execute_external_script
  @language = N'Python',
  @script = N'df=InputDataSet
OutputDataSet=InputDataSet
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Chọn ngẫu nhiên 1000 mẫu để đào tạo và 200 mẫu để kiểm tra
df_train = df.sample(n=800, random_state=42)
df_test = df.sample(n=200, random_state=42)

# Tiền xử lý dữ liệu
tfidf = TfidfVectorizer(stop_words="english")
X_train = tfidf.fit_transform(df_train["text"].values.astype("U"))
y_train = df_train["label"]
X_test = tfidf.transform(df_test["text"].values.astype("U"))
y_test = df_test["label"]

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
,[text] nvarchar(50)
,[label] int))