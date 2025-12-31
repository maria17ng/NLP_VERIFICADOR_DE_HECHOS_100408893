if __name__ == "__main__":
    # ============= RAG TESTING =============
    # from test.test_rag import main
    # main()
    # python test.py

    # ============= INGEST_DATA TESTING =============
    # from ingest.ingest_data import main
    # main()
    # python test.py --clear

    # ============= DB TESTING =============
    # from test.test_db import main
    # main()
    # python test.py --search "¿Cuando se fundó el real madrid?"

    # ============= RETRIEVAL TESTING =============
    # from test.test_retrieval import main
    # main()
    # python test.py

    # ============= VERIFIER TESTING =============
    # from test.test_verifier import test_verifier
    """try:
        success = test_verifier()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        import traceback

        traceback.print_exc()
        exit(1)"""

    """from verifier import FactChecker
    import json

    # Inicializar
    checker = FactChecker()

    # Test simple
    claim_test = "El Real Madrid fue fundado en 1903"

    print("=" * 80)
    print(f"Testing claim: {claim_test}")
    print("=" * 80)

    # Verificar
    result = checker.verify(claim_test)

    print("\n📋 RESULTADO:")
    print(json.dumps(result, indent=2, ensure_ascii=False))"""

    # python test.py

    # from baseline.compare_rag_baseline_table import main
    # main()

    from test.test_multibilingual import test_multilingual
    test_multilingual()
