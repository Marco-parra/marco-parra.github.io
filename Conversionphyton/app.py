from flask import Flask, render_template, request


app = Flask(__name__)




class Token:
    def __init__(self, token_text):
        self.Token = token_text
        self.Mapping = []
        self.SPARQL_Referred_Element = ""
        self.isDataProperty = False
        self.isClass = False
        self.isObjectProperty = False
        self.isIndividual = False

class Translator:
    def __init__(self, language, knowledge_db, ontology):
        pass

    def tokenizar(self, consultaLN):
        # Simulamos tokens de prueba
        t1 = Token("mississippi")
        t1.Mapping.append({"Object":"p#mississippi2","Class":"p#River"})
        t1.Mapping.append({"Object":"p#mississippi_state","Class":"p#State"})

        t2 = Token("area")
        t2.Mapping.append({"Object":"p#area","Class":"p#CDatatypeProperty"})

        return [t1, t2]

    def identificarTokensV2(self, tokens):
        return tokens

    def relacionarTokens(self, tokens):
        return tokens

    def SPARQLQueryGeneration(self, tokens):
        # Devuelve consulta de prueba
        return "SELECT ?x ?y WHERE {?x ?y ?z}"

class OntologyLib:
    def __init__(self, ontology_name, path, prefix):
        pass

    def ExecuteSPARQLQuery(self, sparql_query):
        # Devuelve resultados de prueba
        return [
            {"col1":"Mississippi River", "col2":"River"},
            {"col1":"Mississippi State", "col2":"State"}
        ]

# ---------- END FAKE CLASSES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sparql", methods=["GET"])
def sparql():
    consultaLN = request.args.get("consultaLN", "")
    prueba = request.args.get("prueba")
    indexToken = request.args.get("indexToken")

    oKnowledgeDatabase = OntologyLib(
        "geography.owl",
        "C:\\ILNData\\Ontologies\\",
        "PREFIX p: <http://www.mooney.net/geo#>"
    )

    o = OntologyLib(
        "PRUEBA_alex.OWL",
        "C:\\ILNData\\Ontologies\\",
        "PREFIX p: <http://www.uacj_olap.com/schema/PRUEBA_alex.OWL#>"
    )

    m_Translator = Translator("ENGLISH", oKnowledgeDatabase, o)
    arrTokens = m_Translator.tokenizar(consultaLN)
    arrTokens = m_Translator.identificarTokensV2(arrTokens)
    arrTokens = m_Translator.relacionarTokens(arrTokens)

    arrTokensDialogo = arrTokens

    # Si hubo una selección en el diálogo
    historial = []
    historialIndexTokens = []

    if prueba is not None:
        # Leemos historial de archivo
        historial_path = "C:\\ILNData\\historialSelecciones.txt"
        if os.path.exists(historial_path):
            with open(historial_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        historial.append(int(parts[0]))
                        historialIndexTokens.append(int(parts[1]))

        # Guardamos nueva selección
        historial.append(int(prueba))
        historialIndexTokens.append(int(indexToken))

        # Sobrescribimos historial
        with open(historial_path, "w") as f:
            for h, idx in zip(historial, historialIndexTokens):
                f.write(f"{h} {idx}\n")

        # Aplicar las selecciones al arreglo de tokens
        for j in range(len(historial)):
            index = historial[j]
            indexTok = historialIndexTokens[j]
            token = arrTokensDialogo[indexTok]
            if index < len(token.Mapping):
                mapping = token.Mapping[index]
                tipo = mapping.get("Class", "").upper()
                token.isDataProperty = tipo.endswith("CDATATYPEPROPERTY")
                token.isClass = tipo.endswith("CCLASS")
                token.isObjectProperty = tipo.endswith("COBJECTPROPERTY")
                token.isIndividual = False
            else:
                token.isDataProperty = False
                token.isClass = False
                token.isObjectProperty = False
                token.isIndividual = False

            # Mantener solo el mapping elegido
            for i in range(len(token.Mapping) - 1, -1, -1):
                if i != index:
                    token.Mapping.pop(i)

    elif prueba is None:
        # Si es null, limpiar historial
        with open("C:\\ILNData\\historialSelecciones.txt", "w") as f:
            pass

    # Clasificar tokens que ya solo tienen un mapping
    for token in arrTokensDialogo:
        if len(token.Mapping) == 1:
            mapping = token.Mapping[0]
            if token.isIndividual:
                token.SPARQL_Referred_Element = "INDIVIDUAL"
            elif token.isDataProperty:
                token.SPARQL_Referred_Element = "DATATYPE_PROPERTY"
            elif token.isClass:
                token.SPARQL_Referred_Element = "CLASS"
            elif token.isObjectProperty:
                token.SPARQL_Referred_Element = "OBJECT_PROPERTY"
            else:
                token.SPARQL_Referred_Element = "UNKNOWN"
        elif len(token.Mapping) == 0:
            token.SPARQL_Referred_Element = "NONE"

    resultados = None
    if prueba is not None:
        sparql_query = m_Translator.SPARQLQueryGeneration(arrTokensDialogo)
        resultados = oKnowledgeDatabase.ExecuteSPARQLQuery(sparql_query)

    return render_template("sparql_query_interface.html",
                           consultaLN=consultaLN,
                           arrTokensDialogo=arrTokensDialogo,
                           resultados=resultados)

if __name__ == "__main__":
    app.run(debug=True)
