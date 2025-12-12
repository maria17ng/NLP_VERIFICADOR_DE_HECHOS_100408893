import logging
import sys
from typing import Optional
from pathlib import Path


class ColorCodes:
    """Códigos ANSI para colores en terminal."""

    # Colores básicos
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Colores de texto
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Colores brillantes
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Colores de fondo
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


class ColoredFormatter(logging.Formatter):
    """Formatter personalizado con colores por nivel y clase."""

    # Iconos por nivel de log
    LEVEL_ICONS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🔥'
    }

    # Colores por nivel de log
    LEVEL_COLORS = {
        'DEBUG': ColorCodes.BRIGHT_BLACK,
        'INFO': ColorCodes.BRIGHT_BLUE,
        'WARNING': ColorCodes.BRIGHT_YELLOW,
        'ERROR': ColorCodes.BRIGHT_RED,
        'CRITICAL': ColorCodes.BG_RED + ColorCodes.WHITE
    }

    def __init__(self, class_color: str, class_name: str, fmt: Optional[str] = None):
        """
        Inicializa el formatter con color específico para la clase.

        Args:
            class_color: Código de color ANSI para la clase
            class_name: Nombre de la clase para mostrar
            fmt: Formato del mensaje (opcional)
        """
        super().__init__(fmt or '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.class_color = class_color
        self.class_name = class_name

    def format(self, record):
        """Formatea el mensaje con colores."""
        # Obtener icono y color del nivel
        level_icon = self.LEVEL_ICONS.get(record.levelname, '•')
        level_color = self.LEVEL_COLORS.get(record.levelname, ColorCodes.WHITE)

        # Formatear timestamp
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        timestamp_colored = f"{ColorCodes.DIM}{timestamp}{ColorCodes.RESET}"

        # Formatear nombre de clase
        class_colored = f"{self.class_color}{ColorCodes.BOLD}{self.class_name}{ColorCodes.RESET}"

        # Formatear nivel
        level_colored = f"{level_color}{level_icon} {record.levelname}{ColorCodes.RESET}"

        # Formatear mensaje
        message = record.getMessage()

        # Construir línea completa
        formatted = f"{timestamp_colored} | {class_colored} | {level_colored} | {message}"

        # Si hay excepción, añadirla
        if record.exc_info:
            formatted += '\n' + self.formatException(record.exc_info)

        return formatted


class ModuleColors:
    """Colores asignados a cada módulo del sistema."""

    COLORS = {
        # Core
        'FactChecker': ColorCodes.BRIGHT_MAGENTA,
        'ConfigManager': ColorCodes.BRIGHT_CYAN,

        # Ingesta
        'DocumentIngester': ColorCodes.BRIGHT_GREEN,
        'DocumentPreprocessor': ColorCodes.GREEN,
        'WikipediaPreprocessor': ColorCodes.BRIGHT_GREEN,

        # Chunking
        'SemanticChunker': ColorCodes.BRIGHT_BLUE,
        'HybridChunker': ColorCodes.BLUE,
        'SectionAwareChunker': ColorCodes.CYAN,

        # Metadatos
        'MetadataExtractor': ColorCodes.BRIGHT_YELLOW,
        'ChunkMetadataEnricher': ColorCodes.YELLOW,

        # HyDE
        'HyDEGenerator': ColorCodes.MAGENTA,
        'HyDEAugmenter': ColorCodes.BRIGHT_MAGENTA,
        'SimpleHyDEGenerator': ColorCodes.MAGENTA,

        # Evaluación
        'FactCheckerEvaluator': ColorCodes.BRIGHT_CYAN,

        # Multilingüe
        'ProcesadorMultilingue': ColorCodes.BRIGHT_GREEN,

        # Chat/Usuario
        'Chat': ColorCodes.BRIGHT_WHITE,
        'User': ColorCodes.BRIGHT_WHITE,
        'System': ColorCodes.BRIGHT_BLACK,

        # Default
        'Default': ColorCodes.WHITE
    }

    @classmethod
    def get_color(cls, class_name: str) -> str:
        """
        Obtiene el color para una clase específica.

        Args:
            class_name: Nombre de la clase

        Returns:
            Código de color ANSI
        """
        return cls.COLORS.get(class_name, cls.COLORS['Default'])


def setup_colored_logger(
        class_name: str,
        level: str = 'INFO',
        log_file: Optional[str] = None,
        console: bool = True
) -> logging.Logger:
    """
    Configura un logger con colores para una clase específica.

    Args:
        class_name: Nombre de la clase (determina el color)
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Ruta del archivo de log (opcional)
        console: Si mostrar logs en consola

    Returns:
        Logger configurado
    """
    # Crear logger
    logger = logging.getLogger(class_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Limpiar handlers existentes
    logger.handlers.clear()

    # Obtener color para la clase
    class_color = ModuleColors.get_color(class_name)

    # Handler para consola con colores
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        # Formatter con colores
        console_formatter = ColoredFormatter(
            class_color=class_color,
            class_name=class_name
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Handler para archivo sin colores
    if log_file:
        # Crear directorio si no existe
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))

        # Formatter sin colores para archivo
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Evitar propagación a root logger
    logger.propagate = False

    return logger


def print_separator(char: str = "=", length: int = 70, color: str = ColorCodes.BRIGHT_BLUE):
    """
    Imprime un separador visual con color.

    Args:
        char: Carácter a usar para el separador
        length: Longitud del separador
        color: Color del separador
    """
    print(f"{color}{char * length}{ColorCodes.RESET}")


def print_header(
        text: str,
        color: str = ColorCodes.BRIGHT_CYAN,
        separator: bool = True
):
    """
    Imprime un encabezado destacado con color.

    Args:
        text: Texto del encabezado
        color: Color del encabezado
        separator: Si mostrar separadores arriba y abajo
    """
    if separator:
        print_separator(color=color)

    centered_text = f"{icon}  {text}  {icon}"
    padding = (70 - len(centered_text)) // 2

    print(f"{color}{ColorCodes.BOLD}{' ' * padding}{centered_text}{ColorCodes.RESET}")

    if separator:
        print_separator(color=color)


def demo_loggers():
    """Demuestra el sistema de logging con colores."""
    print_header("DEMO DEL SISTEMA DE LOGGING CON COLORES", icon="🎨")

    # Módulos a demostrar
    modules = [
        'FactChecker',
        'DocumentIngester',
        'SemanticChunker',
        'MetadataExtractor',
        'HyDEGenerator',
        'Chat'
    ]

    print("\n")

    for module in modules:
        logger = setup_colored_logger(module, level='DEBUG', console=True)

        logger.debug(f"Mensaje de debug desde {module}")
        logger.info(f"Mensaje informativo desde {module}")
        logger.warning(f"Advertencia desde {module}")
        logger.error(f"Error desde {module}")

        print()  # Línea en blanco entre módulos

    print_separator()


if __name__ == "__main__":
    # Ejecutar demo
    demo_loggers()
