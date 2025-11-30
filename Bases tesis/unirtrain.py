import pandas as pd

# Cargar los dos CSV
csv1 = pd.read_csv(r"C:\Users\Lenovo\Downloads\dataset2\aptos\train.csv")
csv2 = pd.read_csv(r"C:\Users\Lenovo\Downloads\dataset2\aptos\trainLabels.csv")


# Unirlos
combined = pd.concat([csv1, csv2], ignore_index=True)

# Guardar CSV combinado
combined.to_csv(r"C:\Users\Lenovo\Downloads\dataset2\aptos\trainLabels_total.csv", index=False)
