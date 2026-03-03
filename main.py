import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------- 1. Carregar os Dados -------------------

# Carregar dataset de treino
df = pd.read_csv("treino.csv")

# Separar features (X) e target (y)
X = df.drop(['id', 'target'], axis=1)
y = df['target']

# ------------------- 2. Pré-processamento -------------------

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir treino/validação
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.00025, random_state=42)

# ------------------- 3. Criar a Rede Neural -------------------

# Modelo sequencial simples
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Entrada com número de colunas (13 características)
    layers.Dense(280, activation='relu'),      # Primeira camada oculta
    layers.Dense(280, activation='relu'),      # Segunda camada oculta
    layers.Dense(140, activation='relu'),      # Terceira camada oculta
    layers.Dense(140, activation='softmax')     # Camada de saída para 5 classes
])

# ------------------- 4. Compilar o Modelo -------------------

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',  # Para multi-classe com rótulos inteiros
    metrics=['accuracy']
)

# ------------------- 5. Treinar o Modelo -------------------

# Treinar por 100 épocas (ajustável), com batch_size para acelerar
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=64,
    verbose=1
)

# ------------------- 6. Avaliar o Modelo -------------------

# Avaliar na validação
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\n📊 Acurácia de Validação: {val_accuracy:.4f}")

# ------------------- 7. Predizer Teste -------------------

# Carregar dados de teste
df_test = pd.read_csv("teste.csv")
X_test = df_test.drop(['id'], axis=1)
X_test_scaled = scaler.transform(X_test)

# Realizar predição
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

# ------------------- 8. Salvar o Resultado -------------------

# Criar DataFrame com o formato correto
submission = pd.DataFrame({
    'id': df_test['id'],
    'target': y_pred
})

# Salvar CSV
submission.to_csv('submission.csv', index=False)
print("\n✅ Arquivo 'submission.csv' salvo com sucesso!")