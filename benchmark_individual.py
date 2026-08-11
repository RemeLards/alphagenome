from __future__ import annotations

import argparse
import csv
import os
import subprocess
import threading
from statistics import mean
from time import perf_counter, sleep
from typing import Any, Callable, Dict, List, Optional

import requests

from client import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEQUENCE_LEN,
    AlphaGenomeClient,
    _default_base_url,
    _env_int,
    _format_seconds,
    _load_dotenv,
    build_variant_inputs,
)

DEFAULT_RESULTS_CSV = "individual_vs_batch_benchmark_results.csv"
DEFAULT_NUM_INDIVIDUALS = 8
DEFAULT_POLL_INTERVAL = 0.05
DEFAULT_ROUNDS = 20


def _gpu_memory_used_mb(gpu_index: int) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    value = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    try:
        return float(value)
    except ValueError:
        return None


class GpuMemorySampler:
    def __init__(self, gpu_index: int, interval: float) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self.peak_mb: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self.peak_mb = _gpu_memory_used_mb(self.gpu_index)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[float]:
        self._running = False
        if self._thread is not None:
            self._thread.join()
        current = _gpu_memory_used_mb(self.gpu_index)
        if current is not None:
            self.peak_mb = max(self.peak_mb or current, current)
        return self.peak_mb

    def _sample(self) -> None:
        while self._running:
            current = _gpu_memory_used_mb(self.gpu_index)
            if current is not None:
                self.peak_mb = max(self.peak_mb or current, current)
            sleep(self.interval)


def _format_mb(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f} MB"


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _result_row(
    mode: str,
    round_index: int,
    num_individuals: int,
    total_duration_s: float,
    vram_before_mb: Optional[float],
    vram_peak_mb: Optional[float],
    vram_after_mb: Optional[float],
) -> Dict[str, Any]:
    peak_delta = (
        vram_peak_mb - vram_before_mb
        if vram_peak_mb is not None and vram_before_mb is not None
        else None
    )
    return {
        "mode": mode,
        "round": round_index,
        "num_individuals": num_individuals,
        "total_duration_s": f"{total_duration_s:.6f}",
        "duration_per_individual_s": f"{total_duration_s / num_individuals:.6f}",
        "vram_before_mb": "" if vram_before_mb is None else f"{vram_before_mb:.3f}",
        "vram_peak_mb": "" if vram_peak_mb is None else f"{vram_peak_mb:.3f}",
        "vram_after_mb": "" if vram_after_mb is None else f"{vram_after_mb:.3f}",
        "vram_peak_delta_mb": "" if peak_delta is None else f"{peak_delta:.3f}",
    }


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "round",
        "num_individuals",
        "total_duration_s",
        "duration_per_individual_s",
        "vram_before_mb",
        "vram_peak_mb",
        "vram_after_mb",
        "vram_peak_delta_mb",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _measure_call(
    mode: str,
    round_index: int,
    num_individuals: int,
    gpu_index: int,
    poll_interval: float,
    call: Callable[[], None],
) -> Dict[str, Any]:
    vram_before = _gpu_memory_used_mb(gpu_index)
    sampler = GpuMemorySampler(gpu_index, poll_interval)
    sampler.start()
    start = perf_counter()
    call()
    total_duration = perf_counter() - start
    vram_peak = sampler.stop()
    vram_after = _gpu_memory_used_mb(gpu_index)

    row = _result_row(
        mode=mode,
        round_index=round_index,
        num_individuals=num_individuals,
        total_duration_s=total_duration,
        vram_before_mb=vram_before,
        vram_peak_mb=vram_peak,
        vram_after_mb=vram_after,
    )
    print(
        f"{mode:<10} rodada {round_index:>2}: "
        f"total={_format_seconds(total_duration)} "
        f"por_individuo={_format_seconds(total_duration / num_individuals)} "
        f"vram_pico={_format_mb(vram_peak)} "
        f"delta_pico={_format_mb(float(row['vram_peak_delta_mb']) if row['vram_peak_delta_mb'] else None)}"
    )
    return row


def benchmark_individual_vs_batch(
    client: AlphaGenomeClient,
    intervals: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    ontology_terms: List[str],
    requested_outputs: List[str],
    batch_size: int,
    rounds: int,
    gpu_index: int,
    poll_interval: float,
) -> List[Dict[str, Any]]:
    rows = []
    num_individuals = len(variants)

    def run_individual() -> None:
        for interval, variant in zip(intervals, variants):
            response = client.predict_variant(
                interval=interval,
                variant=variant,
                ontology_terms=ontology_terms,
                requested_outputs=requested_outputs,
            )
            response.get("variant_outputs", [None])[0]

    def run_batch() -> None:
        response = client.predict_variants_batch(
            intervals=intervals,
            variants=variants,
            ontology_terms=ontology_terms,
            requested_outputs=requested_outputs,
            batch_size=batch_size,
        )
        response.get("variant_outputs", [])

    for round_index in range(1, rounds + 1):
        rows.append(
            _measure_call(
                mode="individual",
                round_index=round_index,
                num_individuals=num_individuals,
                gpu_index=gpu_index,
                poll_interval=poll_interval,
                call=run_individual,
            )
        )
        rows.append(
            _measure_call(
                mode="batch",
                round_index=round_index,
                num_individuals=num_individuals,
                gpu_index=gpu_index,
                poll_interval=poll_interval,
                call=run_batch,
            )
        )
    return rows


def _float_values(rows: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(row[key]) for row in rows if row[key] != ""]


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    print("\nResumo")
    print(
        "| modo | rodadas | tempo total medio | tempo medio/individuo | "
        "tempo melhor/individuo | VRAM pico media | VRAM pico maxima | delta VRAM medio |"
    )
    print("|:---|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ("individual", "batch"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        total_times = _float_values(mode_rows, "total_duration_s")
        per_individual_times = _float_values(mode_rows, "duration_per_individual_s")
        vram_peaks = _float_values(mode_rows, "vram_peak_mb")
        vram_deltas = _float_values(mode_rows, "vram_peak_delta_mb")
        print(
            f"| {mode} | {len(mode_rows)} | {_format_seconds(mean(total_times))} | "
            f"{_format_seconds(mean(per_individual_times))} | "
            f"{_format_seconds(min(per_individual_times))} | "
            f"{_format_mb(mean(vram_peaks) if vram_peaks else None)} | "
            f"{_format_mb(max(vram_peaks) if vram_peaks else None)} | "
            f"{_format_mb(mean(vram_deltas) if vram_deltas else None)} |"
        )

    individual_mean = mean(
        _float_values([row for row in rows if row["mode"] == "individual"], "duration_per_individual_s")
    )
    batch_mean = mean(
        _float_values([row for row in rows if row["mode"] == "batch"], "duration_per_individual_s")
    )
    print(f"Speedup medio por individuo do batch: {individual_mean / batch_mean:.2f}x")


def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Compara tempo e VRAM de 8 individuos via individual sequencial vs batch."
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--num-individuals", type=int, default=DEFAULT_NUM_INDIVIDUALS)
    parser.add_argument("--batch-size", type=int, default=_env_int("ALPHAGENOME_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    parser.add_argument("--base-url", default=_default_base_url())
    parser.add_argument("--gpu-index", type=int, default=0, help="Indice da GPU monitorada pelo nvidia-smi.")
    parser.add_argument("--window-size", type=int, default=_env_int("ALPHAGENOME_SEQUENCE_LEN", DEFAULT_SEQUENCE_LEN))
    parser.add_argument("--start", type=int, default=1_000_000)
    parser.add_argument("--ontology-terms", default="UBERON:0001157")
    parser.add_argument("--requested-outputs", default="RNA_SEQ")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--results-csv", default=os.getenv("ALPHAGENOME_INDIVIDUAL_RESULTS_CSV", DEFAULT_RESULTS_CSV))
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    client = AlphaGenomeClient(base_url=args.base_url)
    intervals, variants = build_variant_inputs(
        start=args.start,
        window_size=args.window_size,
        num_variants=args.num_individuals,
    )
    ontology_terms = _parse_csv_list(args.ontology_terms)
    requested_outputs = _parse_csv_list(args.requested_outputs)

    print("--- Configuracao ---")
    print(f"base_url={args.base_url}")
    print(f"gpu_index={args.gpu_index}")
    print(f"window_size={args.window_size}")
    print(f"num_individuals={args.num_individuals}")
    print(f"rounds={args.rounds}")
    print(f"batch_size={args.batch_size}")
    print(f"ontology_terms={','.join(ontology_terms)}")
    print(f"requested_outputs={','.join(requested_outputs)}")

    print("\n--- Warmups descartaveis ---")
    try:
        client.predict_variant(
            interval=intervals[0],
            variant=variants[0],
            ontology_terms=ontology_terms,
            requested_outputs=requested_outputs,
        )
        client.predict_variants_batch(
            intervals=intervals,
            variants=variants,
            ontology_terms=ontology_terms,
            requested_outputs=requested_outputs,
            batch_size=args.batch_size,
        )
    except requests.exceptions.HTTPError as e:
        print(f"Erro HTTP no warmup: {e.response.status_code} - {e.response.text}")
        raise

    print("\n--- Benchmark: individual sequencial vs batch ---")
    rows = benchmark_individual_vs_batch(
        client=client,
        intervals=intervals,
        variants=variants,
        ontology_terms=ontology_terms,
        requested_outputs=requested_outputs,
        batch_size=args.batch_size,
        rounds=args.rounds,
        gpu_index=args.gpu_index,
        poll_interval=args.poll_interval,
    )

    _print_summary(rows)

    if not args.no_csv:
        _write_csv(args.results_csv, rows)
        print(f"CSV salvo em: {args.results_csv}")


if __name__ == "__main__":
    main()
