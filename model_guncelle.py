import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veriyi oku
df = pd.read_csv("heart_guncel.csv")

# Yeni özellik: BMI
df["BMI"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)

# Bağımlı ve bağımsız değişkenleri ayır
X = df.drop("target", axis=1)
y = df["target"]

# Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçekle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeli tanımla ve eğit
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# Test doğruluğunu hesapla
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Seti Doğruluk Oranı: %{acc * 100:.2f}")

# Model ve scaler'ı kaydet
joblib.dump(model, "kalp_modeli.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model ve scaler başarıyla kaydedildi.")
