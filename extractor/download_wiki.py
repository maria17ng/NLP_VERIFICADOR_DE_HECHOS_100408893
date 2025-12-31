import wikipediaapi
import os

# Configuración: Carpeta donde guardarás los textos
DATA_PATH = "data/raw"

# Configuración de Wikipedia (Idioma y User-Agent requerido por la API)
# Es importante poner un user_agent descriptivo para que no bloqueen la petición
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='ProyectoNLP_UC3M_FactChecker (100408893l@alumnos.uc3m.es)',
    language='es',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def download_articles(topics):
    """
    Descarga el contenido de una lista de temas de Wikipedia y los guarda en .txt
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    print(f"⬇️  Iniciando descarga de {len(topics)} artículos...")

    for title in topics:
        page = wiki_wiki.page(title)

        if page.exists():
            print(f"✅ Descargando: {title}")
            
            # Limpiamos el título para que sirva como nombre de archivo
            filename = title.replace(" ", "_").replace("/", "-").replace(':', '-') + ".txt"
            filepath = os.path.join(DATA_PATH, filename)
            
            # Guardamos el texto completo
            with open(filepath, "w", encoding="utf-8") as f:
                # Escribimos la URL original como metadato al principio
                f.write(f"Fuente: {page.fullurl}\n\n")
                f.write(page.text)
        else:
            print(f"⚠️  No existe la página: {title}")

if __name__ == "__main__":
    # LISTA DE TEMAS: Añade aquí lo que quieras que tu sistema "sepa".
    # Para probar verificación de hechos, necesitas temas controvertidos o científicos.
    temas_a_descargar = [
        "Historia del Real Madrid Club de Fútbol",
        "Historia del Club Atlético de Madrid",
        "Anexo:Historia del Club Deportivo Leganés",
        "Anexo:Trayectoria del Getafe Club de Fútbol",
        "Rayo Vallecano de Madrid",
        "Anexo:Palmarés del Real Madrid Club de Fútbol",
        "Anexo:Palmarés del Club Atlético de Madrid"
    ]
    
    download_articles(temas_a_descargar)
