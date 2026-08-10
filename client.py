import argparse
import csv
import os
from pathlib import Path
from time import perf_counter
from typing import List, Dict, Any, Optional

import requests


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_BATCH_SIZE = 2
DEFAULT_SEQUENCE_LEN = 8 * 1000
DEFAULT_WINDOW_SIZES = [
    8 * 1000,
    16 * 1000,
    32 * 1000,
    64 * 1000,
    128 * 1000,
    256 * 1000,
    512 * 1000,
]
DEFAULT_RESULTS_CSV = "window_sweep_results.csv"


def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value)
        os.environ.setdefault(key, value)


def _clean_env_value(value: str) -> str:
    value = value.strip()
    if not value:
        return value

    quote = value[0] if value[0] in {"'", '"'} else ""
    if quote:
        end = value.find(quote, 1)
        return value[1:end] if end != -1 else value[1:]

    return value.split("#", 1)[0].strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _default_base_url() -> str:
    base_url = os.getenv("ALPHAGENOME_BASE_URL")
    if base_url:
        return base_url

    host = os.getenv("ALPHAGENOME_CLIENT_HOST") or os.getenv("ALPHAGENOME_HOST")
    if not host:
        return DEFAULT_BASE_URL
    if host in {"0.0.0.0", "::"}:
        host = "localhost"
    port = _env_int("ALPHAGENOME_PORT", 8000)
    return f"http://{host}:{port}/v1"


def _window_sweep_default() -> bool:
    return _env_bool(
        "ALPHAGENOME_CLIENT_WINDOW_SWEEP",
        _env_bool("ALPHAGENOME_WINDOW_SWEEP"),
    )


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


def _format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def _print_sweep_header() -> None:
    print()
    print("Resumo final")
    print(
        f"{'janela':>8}  {'iters':>5}  {'vars':>4}  {'batch':>5}  "
        f"{'seq melhor':>11}  {'seq media':>10}  "
        f"{'batch melhor':>13}  {'batch media':>11}  "
        f"{'speedup':>8}  status"
    )
    print("-" * 104)


def _print_sweep_row(
    window_size: int,
    rounds: int,
    num_variants: int,
    batch_size: int,
    sequential_times: List[float],
    batch_times: List[float],
    status: str = "ok",
) -> None:
    if not sequential_times or not batch_times:
        print(
            f"{window_size:>8}  {rounds:>5}  {num_variants:>4}  {batch_size:>5}  "
            f"{'-':>11}  {'-':>10}  {'-':>13}  {'-':>11}  {'-':>8}  {status}"
        )
        return

    sequential_best = min(sequential_times)
    sequential_mean = sum(sequential_times) / len(sequential_times)
    batch_best = min(batch_times)
    batch_mean = sum(batch_times) / len(batch_times)
    print(
        f"{window_size:>8}  {rounds:>5}  {num_variants:>4}  {batch_size:>5}  "
        f"{_format_seconds(sequential_best):>11}  {_format_seconds(sequential_mean):>10}  "
        f"{_format_seconds(batch_best):>13}  {_format_seconds(batch_mean):>11}  "
        f"{sequential_mean / batch_mean:>7.2f}x  {status}"
    )


def _sweep_result_row(
    window_size: int,
    rounds: int,
    num_variants: int,
    batch_size: int,
    sequential_times: List[float],
    batch_times: List[float],
    status: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "window_size": window_size,
        "iterations": rounds,
        "num_variants": num_variants,
        "batch_size": batch_size,
        "status": status,
        "sequential_times_s": ";".join(f"{value:.6f}" for value in sequential_times),
        "batch_times_s": ";".join(f"{value:.6f}" for value in batch_times),
    }
    if sequential_times and batch_times:
        sequential_best = min(sequential_times)
        sequential_mean = sum(sequential_times) / len(sequential_times)
        batch_best = min(batch_times)
        batch_mean = sum(batch_times) / len(batch_times)
        row.update(
            {
                "sequential_best_s": f"{sequential_best:.6f}",
                "sequential_mean_s": f"{sequential_mean:.6f}",
                "batch_best_s": f"{batch_best:.6f}",
                "batch_mean_s": f"{batch_mean:.6f}",
                "speedup_mean": f"{sequential_mean / batch_mean:.6f}",
            }
        )
    else:
        row.update(
            {
                "sequential_best_s": "",
                "sequential_mean_s": "",
                "batch_best_s": "",
                "batch_mean_s": "",
                "speedup_mean": "",
            }
        )
    return row


def _write_sweep_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "window_size",
        "iterations",
        "num_variants",
        "batch_size",
        "sequential_best_s",
        "sequential_mean_s",
        "batch_best_s",
        "batch_mean_s",
        "speedup_mean",
        "status",
        "sequential_times_s",
        "batch_times_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_variant_inputs(
    start: int,
    window_size: int,
    num_variants: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    intervals = []
    variants = []
    variant_offset = min(500, max(1, window_size // 2))

    for index in range(num_variants):
        interval_start = start + index * (window_size + 1_000)
        intervals.append(
            {
                "chromosome": "chr1",
                "start": interval_start,
                "end": interval_start + window_size,
            }
        )
        variants.append(
            {
                "chromosome": "chr1",
                "position": interval_start + variant_offset,
                "reference_bases": "A",
                "alternate_bases": "G",
            }
        )

    return intervals, variants


def _normalize_batch_pairs(
    intervals: Any,
    variants: Any,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    interval_list = intervals if isinstance(intervals, list) else [intervals]
    variant_list = variants if isinstance(variants, list) else [variants]

    if len(interval_list) == len(variant_list):
        return interval_list, variant_list
    if len(interval_list) == 1:
        return interval_list * len(variant_list), variant_list
    if len(variant_list) == 1:
        return interval_list, variant_list * len(interval_list)

    raise ValueError(
        "intervals and variants must have the same length, or one side must "
        f"have length 1 for broadcasting: {len(interval_list)} != {len(variant_list)}"
    )


def benchmark_sequential_variants(
    client: "AlphaGenomeClient",
    intervals: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    ontology_terms: List[str],
    requested_outputs: List[str],
    rounds: int,
    print_rounds: bool = False,
) -> tuple[List[float], List[Dict[str, Any]]]:
    times = []
    outputs = []
    for _ in range(rounds):
        round_outputs = []
        start = perf_counter()
        for interval, variant in zip(intervals, variants):
            response = client.predict_variant(
                interval=interval,
                variant=variant,
                ontology_terms=ontology_terms,
                requested_outputs=requested_outputs,
            )
            round_outputs.append(response["variant_outputs"][0])
        times.append(perf_counter() - start)
        if print_rounds:
            print(f"  sequencial rodada {len(times):>2}: {_format_seconds(times[-1])}")
        outputs.extend(round_outputs)
    return times, outputs


def benchmark_batch_variants(
    client: "AlphaGenomeClient",
    intervals: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    ontology_terms: List[str],
    requested_outputs: List[str],
    batch_size: int,
    rounds: int,
    print_rounds: bool = False,
) -> tuple[List[float], List[Dict[str, Any]]]:
    times = []
    outputs = []
    for _ in range(rounds):
        start = perf_counter()
        response = client.predict_variants_batch(
            intervals=intervals,
            variants=variants,
            ontology_terms=ontology_terms,
            requested_outputs=requested_outputs,
            batch_size=batch_size,
        )
        times.append(perf_counter() - start)
        if print_rounds:
            print(f"  batch      rodada {len(times):>2}: {_format_seconds(times[-1])}")
        outputs.extend(response["variant_outputs"])
    return times, outputs


def benchmark_batch_window_sizes(
    client: "AlphaGenomeClient",
    window_sizes: List[int],
    num_variants: int,
    batch_size: int,
    rounds: int,
    start: int,
    ontology_terms: List[str],
    requested_outputs: List[str],
    results_csv: str,
) -> None:
    print("\nWindow sweep: sequencial vs batch")
    print(
        f"rounds={rounds} variantes={num_variants} batch_size={batch_size} "
        f"outputs={','.join(requested_outputs)}"
    )
    rows = []
    csv_rows = []

    for window_size in window_sizes:
        print(f"\nJanela {window_size}")
        intervals, variants = build_variant_inputs(start, window_size, num_variants)
        try:
            print("  warmup sequencial...")
            client.predict_variant(
                interval=intervals[0],
                variant=variants[0],
                ontology_terms=ontology_terms,
                requested_outputs=requested_outputs,
            )
            print("  benchmark sequencial...")
            sequential_times, _ = benchmark_sequential_variants(
                client=client,
                intervals=intervals,
                variants=variants,
                ontology_terms=ontology_terms,
                requested_outputs=requested_outputs,
                rounds=rounds,
                print_rounds=True,
            )

            print("  warmup batch...")
            client.predict_variants_batch(
                intervals=intervals,
                variants=variants,
                ontology_terms=ontology_terms,
                requested_outputs=requested_outputs,
                batch_size=batch_size,
            )
            print("  benchmark batch...")
            batch_times, _ = benchmark_batch_variants(
                client=client,
                intervals=intervals,
                variants=variants,
                ontology_terms=ontology_terms,
                requested_outputs=requested_outputs,
                batch_size=batch_size,
                rounds=rounds,
                print_rounds=True,
            )
            rows.append((window_size, sequential_times, batch_times, "ok"))
            csv_rows.append(
                _sweep_result_row(
                    window_size,
                    rounds,
                    num_variants,
                    batch_size,
                    sequential_times,
                    batch_times,
                    "ok",
                )
            )
        except requests.exceptions.HTTPError as e:
            detail = e.response.text.replace("\n", " ").replace(",", ";")
            status = f"HTTP {e.response.status_code}: {detail}"
            print(f"  erro: {status}")
            rows.append((window_size, [], [], status))
            csv_rows.append(
                _sweep_result_row(
                    window_size,
                    rounds,
                    num_variants,
                    batch_size,
                    [],
                    [],
                    status,
                )
            )

    _print_sweep_header()
    for window_size, sequential_times, batch_times, status in rows:
        _print_sweep_row(
            window_size,
            rounds,
            num_variants,
            batch_size,
            sequential_times,
            batch_times,
            status,
        )
    _write_sweep_csv(results_csv, csv_rows)
    print(f"\nCSV salvo em: {results_csv}")

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
        intervals: Any,
        variants: Any,
        ontology_terms: List[str],
        requested_outputs: List[str],
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Testa o endpoint POST /v1/predict/variants."""
        intervals, variants = _normalize_batch_pairs(intervals, variants)
        if len(intervals) != len(variants):
            raise ValueError(
                "batch payload invalido: "
                f"{len(intervals)} intervals != {len(variants)} variants"
            )
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
    _load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_env_int("ALPHAGENOME_BATCH_SIZE", DEFAULT_BATCH_SIZE),
    )
    parser.add_argument("--num-variants", type=int, default=8)
    parser.add_argument(
        "--window-size",
        type=int,
        default=_env_int("ALPHAGENOME_SEQUENCE_LEN", DEFAULT_SEQUENCE_LEN),
    )
    parser.add_argument("--start", type=int, default=1_000_000)
    parser.add_argument("--base-url", default=_default_base_url())
    parser.add_argument(
        "--batch-only",
        action="store_true",
        default=_env_bool("ALPHAGENOME_CLIENT_BATCH_ONLY"),
        help="Pula warmup/benchmark sequencial e roda direto os testes batch.",
    )
    parser.add_argument(
        "--window-sweep",
        action="store_true",
        default=_window_sweep_default(),
        help="Compara sequencial vs batch para janelas de 8k ate 512k.",
    )
    parser.add_argument(
        "--window-sizes",
        default=os.getenv("ALPHAGENOME_CLIENT_WINDOW_SIZES"),
        help="Lista de janelas separadas por virgula. Default: 8000,...,512000.",
    )
    parser.add_argument(
        "--results-csv",
        default=os.getenv("ALPHAGENOME_CLIENT_RESULTS_CSV", DEFAULT_RESULTS_CSV),
        help="Arquivo CSV para salvar o resumo do window sweep.",
    )
    args = parser.parse_args()

    # Ajuste para a URL onde seu servidor FastAPI está rodando
    client = AlphaGenomeClient(base_url=args.base_url)

    sample_terms = ["UBERON:0001157"]
    sample_outputs = ["RNA_SEQ"]

    if args.window_sweep:
        window_sizes = (
            _parse_int_list(args.window_sizes)
            if args.window_sizes
            else DEFAULT_WINDOW_SIZES
        )
        benchmark_batch_window_sizes(
            client=client,
            window_sizes=window_sizes,
            num_variants=args.num_variants,
            batch_size=args.batch_size,
            rounds=args.rounds,
            start=args.start,
            ontology_terms=sample_terms,
            requested_outputs=sample_outputs,
            results_csv=args.results_csv,
        )
        raise SystemExit(0)

    intervals, variants = build_variant_inputs(
        start=args.start,
        window_size=args.window_size,
        num_variants=args.num_variants,
    )
    sample_interval = intervals[0]
    sample_variant = variants[0]
    rounds = args.rounds
    batch_size = args.batch_size

    sequential_times = []
    sequential_outputs = []
    if args.batch_only:
        print("--- Pulando chamadas sequenciais (--batch-only) ---")
    else:
        print("--- Warmup sequencial descartavel ---")
        client.predict_variant(
            interval=sample_interval,
            variant=sample_variant,
            ontology_terms=sample_terms,
            requested_outputs=sample_outputs,
        )

        print(f"--- Benchmark: {args.num_variants} chamadas sequenciais ---")
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

    print(
        f"--- Benchmark: 1 chamada HTTP com {args.num_variants} variantes "
        f"(batch_size={batch_size}) ---"
    )
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

    batch_best = min(batch_times)
    batch_mean = sum(batch_times) / len(batch_times)
    print("\nResumo")
    print("| modo | melhor | media |")
    print("|:---|---:|---:|")
    print(f"| batch | {_format_seconds(batch_best)} | {_format_seconds(batch_mean)} |")
    if sequential_times:
        sequential_best = min(sequential_times)
        sequential_mean = sum(sequential_times) / len(sequential_times)
        print(
            f"| sequencial | {_format_seconds(sequential_best)} | "
            f"{_format_seconds(sequential_mean)} |"
        )
        print(f"Speedup medio batch vs sequencial: {sequential_mean / batch_mean:.2f}x")

    if args.batch_only:
        raise SystemExit(0)

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
