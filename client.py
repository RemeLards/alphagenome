import requests
from typing import List, Dict, Any, Optional

class AlphaGenomeClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> Dict[str, Any]:
        """Testa o endpoint GET /v1/health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def predict_variant(
        self,
        interval: Dict[str, Any],
        variant: Dict[str, Any],
        ontology_terms: List[str],
        requested_outputs: List[str]
    ) -> Dict[str, Any]:
        """Testa o endpoint POST /v1/predict/variant."""
        payload = {
            "interval": interval,
            "variant": variant,
            "ontology_terms": ontology_terms,
            "requested_outputs": requested_outputs
        }
        response = requests.post(f"{self.base_url}/predict/variant", json=payload)
        response.raise_for_status()
        return response.status_code

    def predict_variants_batch(
        self,
        intervals: List[Dict[str, Any]],
        variants: List[Dict[str, Any]],
        ontology_terms: List[str],
        requested_outputs: List[str]
    ) -> Dict[str, Any]:
        """Testa o endpoint POST /v1/predict/variants."""
        payload = {
            "intervals": intervals,
            "variants": variants,
            "ontology_terms": ontology_terms,
            "requested_outputs": requested_outputs
        }
        response = requests.post(f"{self.base_url}/predict/variants", json=payload)
        response.raise_for_status()
        return response.status_code

    def predict_interval(
        self,
        interval: Dict[str, Any],
        ontology_terms: List[str],
        requested_outputs: List[str]
    ) -> Dict[str, Any]:
        """Testa o endpoint POST /v1/predict/interval."""
        payload = {
            "interval": interval,
            "ontology_terms": ontology_terms,
            "requested_outputs": requested_outputs
        }
        response = requests.post(f"{self.base_url}/predict/interval", json=payload)
        response.raise_for_status()
        return response.json()


# ==============================================================================
# Script de Testes
# ==============================================================================
if __name__ == "__main__":
    # Ajuste para a URL onde seu servidor FastAPI está rodando
    client = AlphaGenomeClient(base_url="http://localhost:8000/v1")

    # Exemplo de dados (ajuste os campos para corresponder aos seus Pydantic Schemas)
    sample_interval = {
        "chromosome": "chr1",  # era "chr"
        "start": 1000,
        "end": 1000 + 8 * 1024,
    }

    # 2. Ajuste da Variante
    sample_variant = {
        "chromosome": "chr1",  # era "chr"
        "position": 1500,  # era "pos"
        "reference_bases": "A",  # era "ref"
        "alternate_bases": "G",  # era "alt"
    }

    sample_terms = ["UBERON:0001157"]
    sample_outputs = ["RNA_SEQ", "ATAC"]

    # print("--- 1. Testando Health Check ---")
    # try:
    #     health_info = client.check_health()
    #     print(f"Health Status: {health_info}\n")
    # except Exception as e:
    #     print(f"Erro no health check: {e}\n")

    print("--- 2. Testando Predict Variant ---")
    try:
        variant_res = client.predict_variant(
            interval=sample_interval,
            variant=sample_variant,
            ontology_terms=sample_terms,
            requested_outputs=sample_outputs
        )
        print(f"Predict Variant Resultado: {variant_res}\n")
    except requests.exceptions.HTTPError as e:
        print(f"Erro HTTP em Predict Variant: {e.response.status_code} - {e.response.text}\n")

    print("--- 3. Testando Predict Variants (Batch) ---")
    try:
        batch_res = client.predict_variants_batch(
            intervals=[sample_interval],
            variants=[sample_variant],
            ontology_terms=sample_terms,
            requested_outputs=sample_outputs
        )
        print(f"Predict Variants Batch Resultado: {batch_res}\n")
    except requests.exceptions.HTTPError as e:
        print(f"Erro HTTP em Predict Batch: {e.response.status_code} - {e.response.text}\n")

    # print("--- 4. Testando Predict Interval ---")
    # try:
    #     interval_res = client.predict_interval(
    #         interval=sample_interval,
    #         ontology_terms=sample_terms,
    #         requested_outputs=sample_outputs
    #     )
    #     print(f"Predict Interval Resultado: {interval_res}\n")
    # except requests.exceptions.HTTPError as e:
    #     print(f"Erro HTTP em Predict Interval: {e.response.status_code} - {e.response.text}\n")