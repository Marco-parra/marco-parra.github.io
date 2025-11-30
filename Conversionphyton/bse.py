import sqlite3

# Conexión a la base de datos
conn = sqlite3.connect("mi_sistema.db")
cursor = conn.cursor()

# Crear tabla (si no existe)
cursor.execute("""
CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT,
    edad INTEGER,
    diagnostico TEXT
)
""")

# Insertar un registro de ejemplo
cursor.execute("INSERT INTO pacientes (nombre, edad, diagnostico) VALUES (?, ?, ?)",
               ("Juan Pérez", 60, "Retinopatía diabética"))
conn.commit()

# Simulación de lenguaje natural
pregunta = "Muéstrame los pacientes con retinopatía diabética"

if "retinopatía diabética" in pregunta.lower():
    sql = "SELECT * FROM pacientes WHERE diagnostico = 'Retinopatía diabética'"

cursor.execute(sql)
resultados = cursor.fetchall()

print("Resultados:")
for fila in resultados:
    print(fila)

conn.close()
