--------------------Instrucciones----------------------------
🚀 Requisitos
Antes de iniciar, asegúrate de tener instalado:

Python 3.9+

pip

Flask

rdflib

Instalar dependencias

pip install flask rdflib

▶️ Ejecutar el servidor
Ejecuta el proyecto con:

python app.py

Deberías ver algo como:

Running on http://127.0.0.1:5000/

Abre el navegador y ve a:

http://127.0.0.1:5000

📂 Estructura de archivos

proyecto/
│
├── app.py                  # Archivo principal de Flask
├── data.rdf                # Archivo RDF con los datos
├── templates/
│   └── sparql_query_interface.html  # Interfaz web
└── README.md               # Documentación del Proyecto

🖥 Uso de la API
Para ejecutar una consulta SPARQL desde el navegador:

http://127.0.0.1:5000/sparql?consultaLN=SELECT+*+WHERE+{?s+?p+?o}+LIMIT+10

📌 Notas
Si obtienes el error Graph not defined, asegúrate de importar:

from rdflib import Graph