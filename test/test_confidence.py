"""
Test del sistema de confianza mejorado con similitud semántica.
"""

from verifier import FactChecker


def test_confidence_system():
    """Prueba el sistema de confianza con diferentes claims."""

    print("=" * 80)
    print("TEST DEL SISTEMA DE CONFIANZA MEJORADO")
    print("=" * 80)

    # Inicializar verificador
    checker = FactChecker(config_path="settings/config.yaml")

    # Test 1: Claim con alta similitud (fecha correcta)
    print("\n" + "=" * 80)
    print("TEST 1: Alta confianza - Fecha correcta con evidencia clara")
    print("=" * 80)
    result1 = checker.verify("El Real Madrid fue fundado en 1902")
    print(f"\n📊 Veredicto: {result1['veredicto']}")
    print(f"🎯 Confianza: {result1['nivel_confianza']}/5")
    print(f"💬 Explicación: {result1['explicacion_corta']}")

    # Test 2: Claim con fecha incorrecta
    print("\n" + "=" * 80)
    print("TEST 2: Media confianza - Fecha incorrecta con contradicción clara")
    print("=" * 80)
    result2 = checker.verify("El Real Madrid fue fundado en 1903")
    print(f"\n📊 Veredicto: {result2['veredicto']}")
    print(f"🎯 Confianza: {result2['nivel_confianza']}/5")
    print(f"💬 Explicación: {result2['explicacion_corta']}")

    # Test 3: Betis sin evidencia
    print("\n" + "=" * 80)
    print("TEST 3: Baja confianza - Sin evidencia relevante")
    print("=" * 80)
    result3 = checker.verify("El Real Betis fue fundado en 1907")
    print(f"\n📊 Veredicto: {result3['veredicto']}")
    print(f"🎯 Confianza: {result3['nivel_confianza']}/5")
    print(f"💬 Explicación: {result3['explicacion_corta']}")

    # Test 4: Champions League (evidencia fuerte)
    print("\n" + "=" * 80)
    print("TEST 4: Alta confianza - Claim genérico con mucha evidencia")
    print("=" * 80)
    result4 = checker.verify("El Real Madrid es un club de fútbol español")
    print(f"\n📊 Veredicto: {result4['veredicto']}")
    print(f"🎯 Confianza: {result4['nivel_confianza']}/5")
    print(f"💬 Explicación: {result4['explicacion_corta']}")

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE CONFIANZA")
    print("=" * 80)

    results = [
        ("RM 1902 (correcto)", result1['nivel_confianza']),
        ("RM 1903 (incorrecto)", result2['nivel_confianza']),
        ("Betis 1907 (sin evidencia)", result3['nivel_confianza']),
        ("RM es español", result4['nivel_confianza'])
    ]

    for name, conf in results:
        bar = "█" * conf + "░" * (5 - conf)
        print(f"  {name:30} [{bar}] {conf}/5")

    print("\n✅ Tests de confianza completados")


if __name__ == "__main__":
    test_confidence_system()
