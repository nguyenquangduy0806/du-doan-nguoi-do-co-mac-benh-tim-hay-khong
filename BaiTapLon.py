import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

data_path = 'd:\TRITUENHANTAO/heart_disease_uci (2).csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Không tìm thấy tệp {data_path}. Hãy kiểm tra lại!")

# Đọc dữ liệu
df = pd.read_csv(data_path)

# Hiển thị thông tin dữ liệu
df.info()
print(df.describe())

# Kiểm tra tên cột
print("Tên các cột trong dữ liệu:", df.columns)

target_column = 'num'  # Đổi từ 'target' thành 'num'

# Kiểm tra xem cột mục tiêu có tồn tại không
if target_column not in df.columns:
    raise KeyError(f"Cột '{target_column}' không tồn tại trong dữ liệu. Hãy kiểm tra lại!")

# Chuyển đổi bài toán về nhị phân
df[target_column] = (df[target_column] > 0).astype(int)

# Kiểm tra giá trị của cột mục tiêu
print(f"Giá trị duy nhất trong cột '{target_column}':", df[target_column].unique())

# Kiểm tra giá trị thiếu
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Xử lý giá trị thiếu bằng cách điền trung vị cho các cột số
df = df.fillna(df.median(numeric_only=True))

# Loại bỏ các cột không cần thiết (ví dụ: ID, dataset name nếu có)
categorical_columns = df.select_dtypes(include=['object']).columns
df = df.drop(columns=categorical_columns)

# Tách đặc trưng (X) và nhãn (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Xây dựng mô hình ANN bằng sklearn
model = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)
print("Báo cáo phân loại:\n", classification_report(y_test, y_pred))

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Không bệnh', 'Bị bệnh'], yticklabels=['Không bệnh', 'Bị bệnh'])
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn")
plt.show()

# Hiển thị loss curve
plt.plot(model.loss_curve_)
plt.xlabel("Số lần lặp")
plt.ylabel("Mất mát")
plt.title("Biểu đồ mất mát")
plt.show()

# Hiển thị dự đoán từng cá nhân
predictions_df = pd.DataFrame(X_test, columns=df.drop(columns=[target_column]).columns)
predictions_df['Thực tế'] = y_test.values
predictions_df['Dự đoán'] = y_pred
predictions_df['Kết quả dự đoán'] = predictions_df['Dự đoán'].apply(lambda x: 'Bị bệnh' if x == 1 else 'Không bệnh')

print("\nDự đoán cho từng cá nhân:\n")
print(predictions_df.head(10))