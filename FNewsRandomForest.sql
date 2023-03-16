EXEC sp_execute_external_script
  @language = N'Python',
  @script = N'df=InputDataSet
OutputDataSet=InputDataSet
# Import các thư viện cần thiết
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Tiền xử lý dữ liệu
tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(df["text"].values.astype("U"))
y = df["label"]

# Phân chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Xây dựng mô hình Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Đào tạo mô hình Random Forest
rf.fit(X_train, y_train)

# Đánh giá mô hình Random Forest
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của mô hình Random Forest:", accuracy)


',
@input_data_1=N'SELECT top (1000) * from dbo.FakeNews'
with result sets((
[column1] int
,[title] nvarchar(50)
,[text] nvarchar(50)
,[label] int))