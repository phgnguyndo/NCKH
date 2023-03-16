
EXEC sp_execute_external_script
    @language = N'Python',
    @script = N'df=InputDataSet
OutputDataSet=InputDataSet
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Xử lý giá trị NaN
df["text"].fillna("", inplace=True)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)

# Tạo vector đặc trưng với TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)

# Biến đổi dữ liệu văn bản thành ma trận vector đặc trưng
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Huấn luyện mô hình Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
score = clf.score(X_test, y_test)
print("Accuracy: ", score)
',
@input_data_1=N'select top (1000) * from dbo.FakeNews '
with result sets((
[column1] int
,[title] nvarchar(50)
,[text] NVARCHAR(50)
,[label] int))
GO

SELECT * FROM dbo.FakeNews