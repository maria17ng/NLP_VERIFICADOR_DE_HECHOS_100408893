"""
Módulo de utilidades para el sistema de verificación de hechos.
Incluye funciones para logging, carga de configuración y métricas.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from logger.logger import setup_colored_logger


def setup_logger(
    name: str = "FactChecker",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Configura y retorna un logger.

    Args:
        name: Nombre del logger
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Ruta del archivo de log (opcional)
        console: Si mostrar logs en consola

    Returns:
        Logger configurado
    """

    return setup_colored_logger(class_name=name, level=level, log_file=log_file, console=console)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración

    Returns:
        Diccionario con la configuración

    Raises:
        FileNotFoundError: Si no se encuentra el archivo de configuración
    """
    # Intentar diferentes rutas
    possible_paths = [
        config_path,
        f"settings/{config_path}",
        os.path.join(os.path.dirname(__file__), config_path)
    ]

    config_file = None
    for path in possible_paths:
        if os.path.exists(path):
            config_file = path
            break

    if not config_file:
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración. Rutas intentadas: {possible_paths}"
        )

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def load_prompts(prompts_path: str) -> Dict[str, str]:
    """
    Carga los prompts desde un archivo YAML.

    Args:
        prompts_path: Ruta al archivo de prompts

    Returns:
        Diccionario con los prompts
    """
    # Intentar diferentes rutas
    possible_paths = [
        prompts_path,
        f"verificador/{prompts_path}",
        os.path.join(os.path.dirname(__file__), prompts_path)
    ]

    prompts_file = None
    for path in possible_paths:
        if os.path.exists(path):
            prompts_file = path
            break

    if not prompts_file:
        raise FileNotFoundError(
            f"No se encontró el archivo de prompts en: {prompts_path}"
        )

    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)

    return prompts


def ensure_directory(directory: str) -> None:
    """
    Crea un directorio si no existe.

    Args:
        directory: Ruta del directorio a crear
    """
    os.makedirs(directory, exist_ok=True)


def get_project_root() -> Path:
    """
    Obtiene la ruta raíz del proyecto.

    Returns:
        Path object con la ruta raíz
    """
    return Path(__file__).parent


def resolve_path(path: str, create_if_missing: bool = False) -> str:
    """
    Resuelve una ruta relativa o absoluta, intentando diferentes ubicaciones.

    Args:
        path: Ruta a resolver
        create_if_missing: Si crear el directorio si no existe

    Returns:
        Ruta resuelta
    """
    # Si es absoluta y existe, devolverla
    if os.path.isabs(path) and os.path.exists(path):
        return path

    # Intentar rutas relativas
    possible_paths = [
        path,
        os.path.join("verificador", path),
        os.path.join(get_project_root(), path)
    ]

    for p in possible_paths:
        if os.path.exists(p):
            return p

    # Si no existe y se debe crear
    if create_if_missing:
        # Usar la primera opción como default
        ensure_directory(os.path.dirname(possible_paths[0]))
        return possible_paths[0]

    # Devolver la ruta original si no se encuentra
    return path


class ConfigManager:
    """
    Gestor centralizado de configuración para el sistema.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el gestor de configuración.

        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self._setup_paths()

    def _setup_paths(self):
        """Configura las rutas del sistema."""
        paths = self.config.get('paths', {})
        for key, path in paths.items():
            # Crear directorios si no existen
            if key in ['logs', 'vector_store', 'evaluations', 'data_raw']:
                ensure_directory(path)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de la configuración usando notación de punto.

        Args:
            key: Clave en formato 'section.subsection.value'
            default: Valor por defecto si no se encuentra

        Returns:
            Valor de configuración

        Example:
            >>> config.get('models.llm.temperature')
            0.1
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_path(self, key: str) -> str:
        """
        Obtiene una ruta de la configuración.

        Args:
            key: Nombre de la ruta (e.g., 'vector_store', 'prompts')

        Returns:
            Ruta resuelta
        """
        path = self.config['paths'].get(key, '')
        return resolve_path(path)

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Obtiene la configuración de un modelo específico.

        Args:
            model_type: Tipo de modelo ('llm', 'embeddings', 'reranker')

        Returns:
            Diccionario con la configuración del modelo
        """
        return self.config.get('models', {}).get(model_type, {})

    def get_rag_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración del sistema RAG.

        Returns:
            Diccionario con la configuración RAG
        """
        return self.config.get('rag', {})


if __name__ == "__main__":
    # Ejemplo de uso
    logger = setup_logger(level="DEBUG", log_file="logs/test.log")
    logger.info("Logger configurado correctamente")

    try:
        config = ConfigManager()
        print(f"Temperatura LLM: {config.get('models.llm.temperature')}")
        print(f"Ruta vector store: {config.get_path('vector_store')}")
        print(f"Configuración RAG: {config.get_rag_config()}")
    except FileNotFoundError as e:
        logger.error(f"Error cargando configuración: {e}")
