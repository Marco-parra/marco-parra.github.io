import pandas as pd

# Ruta del archivo CSV original
csv_path = r"C:\Users\Lenovo\Downloads\archive\train.csv"

# Leer el CSV
df = pd.read_csv(csv_path)

# Quitar ceros a la izquierda en la columna 'id_code'
df["id_code"] = df["id_code"].astype(str).str.lstrip("0")

# Guardar el resultado en un nuevo archivo
output_path = r"C:\Users\Lenovo\Downloads\archive\train_sin_ceros.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ Archivo guardado sin ceros en: {output_path}")
