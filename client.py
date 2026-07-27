import requests
from typing import List, Dict, Any, Optional
from time import perf_counter
import argparse


def _flatten_numbers(value: Any) -> List[float]:
    if isinstance(value, dict):
        numbers: List[float] = []
        for item in value.values():
            numbers.extend(_flatten_numbers(item))
        return numbers
    if isinstance(value, list):
        numbers: List[float] = []
        for item in value:
            numbers.extend(_flatten_numbers(item))
        return numbers
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def _diff_metrics(left_values: List[float], right_values: List[float]) -> Dict[str, float]:
    if len(left_values) != len(right_values):
        raise ValueError(
            f"outputs have different numeric sizes: {len(left_values)} != {len(right_values)}"
        )

    diffs = [abs(a - b) for a, b in zip(left_values, right_values)]
    max_abs = max(diffs, default=0.0)
    mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
    rms = (sum(diff * diff for diff in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
    max_ref = max((abs(v) for v in left_values), default=0.0)
    return {
        "count": float(len(diffs)),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rms": rms,
        "max_abs_relative_to_ref": max_abs / max_ref if max_ref else 0.0,
    }


def _variant_values(output: Dict[str, Any]) -> List[float]:
    all_left_values: List[float] = []

    for allele in ("reference", "alternate"):
        for track in output.get(allele, {}).values():
            if track is not None:
                all_left_values.extend(_flatten_numbers(track.get("values", [])))
    return all_left_values


def compare_variant_outputs(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, float]:
    return _diff_metrics(_variant_values(left), _variant_values(right))


def _nested_shape(value: Any) -> List[int]:
    shape = []
    while isinstance(value, list):
        shape.append(len(value))
        value = value[0] if value else []
    return shape


def describe_variant_output(output: Dict[str, Any]) -> None:
    total = 0
    for allele in ("reference", "alternate"):
        for output_name, track in output.get(allele, {}).items():
            if track is None:
                continue
            values = track.get("values", [])
            count = len(_flatten_numbers(values))
            total += count
            print(
                f"{allele}.{output_name}: shape={_nested_shape(values)} "
                f"values={count}"
            )
    print(f"Total values por variante: {total}")

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
        return response.json()

    def predict_variants_batch(
        self,
        intervals: List[Dict[str, Any]],
        variants: List[Dict[str, Any]],
        ontology_terms: List[str],
        requested_outputs: List[str],
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Testa o endpoint POST /v1/predict/variants."""
        payload = {
            "intervals": intervals,
            "variants": variants,
            "ontology_terms": ontology_terms,
            "requested_outputs": requested_outputs
        }
        if batch_size is not None:
            payload["batch_size"] = batch_size
        response = requests.post(f"{self.base_url}/predict/variants", json=payload)
        response.raise_for_status()
        return response.json()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Ajuste para a URL onde seu servidor FastAPI está rodando
    client = AlphaGenomeClient(base_url="http://localhost:8000/v1")

    # Exemplo de dados (ajuste os campos para corresponder aos seus Pydantic Schemas)
    sample_interval = {
        "chromosome": "chr1",  # era "chr"
        "start": 1_000_000,
        "end": 1_000_000 + 8 * 1024,
    }

    # 2. Ajuste da Variante
    sample_variant = {
        "chromosome": "chr1",  # era "chr"
        "position": 1_000_500,  # era "pos"
        "reference_bases": "A",  # era "ref"
        "alternate_bases": "G",  # era "alt"
    }

    sample_terms = ["UBERON:0001157"]
    sample_outputs = ["RNA_SEQ"]

    # print("--- 1. Testando Health Check ---")
    # try:
    #     health_info = client.check_health()
    #     print(f"Health Status: {health_info}\n")
    # except Exception as e:
    #     print(f"Erro no health check: {e}\n")

    intervals = [sample_interval] * 8
    variants = [sample_variant] * 8
    rounds = args.rounds
    batch_size = args.batch_size

    print("--- Warmup sequencial descartavel ---")
    client.predict_variant(
        interval=sample_interval,
        variant=sample_variant,
        ontology_terms=sample_terms,
        requested_outputs=sample_outputs,
    )

    print("--- Benchmark: 8 chamadas sequenciais ---")
    sequential_times = []
    sequential_outputs = []
    try:
        for round_index in range(rounds):
            round_outputs = []
            start = perf_counter()
            for interval, variant in zip(intervals, variants):
                response = client.predict_variant(
                    interval=interval,
                    variant=variant,
                    ontology_terms=sample_terms,
                    requested_outputs=sample_outputs,
                )
                round_outputs.append(response["variant_outputs"][0])
            sequential_time = perf_counter() - start
            sequential_times.append(sequential_time)
            sequential_outputs.extend(round_outputs)
            print(f"Rodada {round_index + 1}: {sequential_time:.3f}s")
    except requests.exceptions.HTTPError as e:
        print(f"Erro HTTP no sequencial: {e.response.status_code} - {e.response.text}\n")
        raise

    print(f"--- Warmup batch descartavel (batch_size={batch_size}) ---")
    client.predict_variants_batch(
        intervals=intervals,
        variants=variants,
        ontology_terms=sample_terms,
        requested_outputs=sample_outputs,
        batch_size=batch_size,
    )

    print(f"--- Benchmark: 1 chamada HTTP com 8 variantes (batch_size={batch_size}) ---")
    batch_times = []
    batch_outputs = []
    try:
        for round_index in range(rounds):
            start = perf_counter()
            batch_res = client.predict_variants_batch(
                intervals=intervals,
                variants=variants,
                ontology_terms=sample_terms,
                requested_outputs=sample_outputs,
                batch_size=batch_size,
            )
            batch_time = perf_counter() - start
            batch_times.append(batch_time)
            batch_outputs.extend(batch_res["variant_outputs"])
            print(f"Rodada {round_index + 1}: {batch_time:.3f}s")
    except requests.exceptions.HTTPError as e:
        print(f"Erro HTTP no batch: {e.response.status_code} - {e.response.text}\n")
        raise

    sequential_best = min(sequential_times)
    batch_best = min(batch_times)
    sequential_mean = sum(sequential_times) / len(sequential_times)
    batch_mean = sum(batch_times) / len(batch_times)
    print("--- Resumo de velocidade ---")
    print(f"Sequencial melhor: {sequential_best:.3f}s")
    print(f"Batch melhor: {batch_best:.3f}s")
    print(f"Speedup melhor: {sequential_best / batch_best:.2f}x")
    print(f"Sequencial media: {sequential_mean:.3f}s")
    print(f"Batch media: {batch_mean:.3f}s")
    print(f"Speedup medio: {sequential_mean / batch_mean:.2f}x")

    print("--- Diferença entre outputs ---")
    print("--- Dimensões comparadas ---")
    describe_variant_output(sequential_outputs[0])
    comparisons = [
        compare_variant_outputs(sequential_output, batch_output)
        for sequential_output, batch_output in zip(sequential_outputs, batch_outputs)
    ]
    values_per_comparison = int(comparisons[0]["count"])
    total_values = values_per_comparison * len(comparisons)
    max_abs = max(item["max_abs"] for item in comparisons)
    mean_abs = sum(item["mean_abs"] for item in comparisons) / len(comparisons)
    rms = sum(item["rms"] for item in comparisons) / len(comparisons)
    max_rel = max(item["max_abs_relative_to_ref"] for item in comparisons)
    print(f"Comparações: {len(comparisons)}")
    print(f"Values por comparação: {values_per_comparison}")
    print(f"Values totais comparados: {total_values}")
    print(f"Max abs diff: {max_abs:.8g}")
    print(f"Mean abs diff: {mean_abs:.8g}")
    print(f"Mean RMS diff: {rms:.8g}")
    print(f"Max relative diff: {max_rel:.8g}")

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
