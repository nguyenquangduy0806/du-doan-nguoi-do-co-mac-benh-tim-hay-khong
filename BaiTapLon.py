import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Đọc dữ liệu
data_path = "d:\TRITUENHANTAO/heart_disease_uci.csv"
df = pd.read_csv(data_path)

# Hiển thị thông tin dữ liệu
print("Thông tin dữ liệu ban đầu:")
print(df.info())
print(df.describe())
print(df.head())

# Kiểm tra giá trị thiếu
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table.columns = ['Số lượng thiếu', 'Phần trăm (%)']
    return mis_val_table[mis_val_table.iloc[:, 0] > 0]

print("Giá trị thiếu:")
print(missing_values_table(df))

# Xử lý giá trị thiếu (nếu có)
df.fillna(df.median(), inplace=True)

# Mã hóa dữ liệu dạng phân loại (nếu có)
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Giả sử cột cuối là nhãn bệnh tim (1: mắc bệnh, 0: không mắc)
X = df.iloc[:, :-1]  # Các thuộc tính đầu vào
y = df.iloc[:, -1]    # Nhãn dự đoán

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Xây dựng mô hình mạng nơ-ron nhân tạo
model = MLPClassifier(hidden_layer_sizes=(32, 16, 8), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Tìm kiếm tham số tốt nhất
param_grid = {
    'hidden_layer_sizes': [(16, 8), (32, 16, 8), (64, 32, 16)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [500, 1000]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Hiển thị tham số tốt nhất
print("Best parameters found:", grid_search.best_params_)
model = grid_search.best_estimator_

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Hiển thị kết quả đánh giá
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy:.4f}")
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred))

# Vẽ ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Không mắc bệnh', 'Mắc bệnh'], yticklabels=['Không mắc bệnh', 'Mắc bệnh'])
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn")
plt.show()

# Xuất bảng dữ liệu dự đoán
predictions_df = pd.DataFrame({'Thực tế': y_test.values, 'Dự đoán': y_pred})
print("Bảng dữ liệu dự đoán:")
print(predictions_df.head())