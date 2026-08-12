from __future__ import annotations

import argparse
import csv
import os
import re
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
DEFAULT_NUM_INDIVIDUALS = 4
DEFAULT_BATCH_SIZE = 8
DEFAULT_STRANDS = "+,-"
DEFAULT_ONTOLOGY_TERMS = "UBERON:0001157,UBERON:0002107,UBERON:0002048"
DEFAULT_POLL_INTERVAL = 0.05
DEFAULT_ROUNDS = 10
_NVIDIA_SMI_WARNING_PRINTED = False


def _warn_nvidia_smi(message: str) -> None:
    global _NVIDIA_SMI_WARNING_PRINTED
    if not _NVIDIA_SMI_WARNING_PRINTED:
        print(f"Aviso: nao foi possivel medir VRAM via nvidia-smi: {message}")
        _NVIDIA_SMI_WARNING_PRINTED = True


def _gpu_memory_used_mb(gpu_index: int, process_pid: Optional[int]) -> Optional[float]:
    value = None if process_pid is not None else _gpu_memory_used_mb_from_query(gpu_index)
    if value is not None:
        return value
    return _gpu_memory_used_mb_from_table(gpu_index, process_pid)


def _gpu_memory_used_mb_from_query(gpu_index: int) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        _warn_nvidia_smi("comando nvidia-smi nao encontrado no PATH")
        return None
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or str(e)).strip()
        _warn_nvidia_smi(detail)
        return None

    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        if parts[0] != str(gpu_index):
            continue
        try:
            return float(parts[1])
        except ValueError:
            return None

    return None


def _gpu_memory_used_mb_from_table(
    gpu_index: int,
    process_pid: Optional[int],
) -> Optional[float]:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        _warn_nvidia_smi("comando nvidia-smi nao encontrado no PATH")
        return None
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or str(e)).strip()
        _warn_nvidia_smi(detail)
        return None

    lines = result.stdout.splitlines()
    gpu_header_pattern = re.compile(rf"^\|\s*{gpu_index}\s+")
    memory_pattern = re.compile(r"(\d+)\s*MiB\s*/\s*\d+\s*MiB")
    process_pattern = re.compile(r"^\|\s*(\d+)\s+\S+\s+\S+\s+(\d+)\s+.*?\s+(\d+)\s*MiB\s*\|")

    if process_pid is None:
        for index, line in enumerate(lines):
            if not gpu_header_pattern.search(line):
                continue
            for candidate in lines[index : index + 4]:
                match = memory_pattern.search(candidate)
                if match:
                    return float(match.group(1))

    process_total = 0.0
    matched_process = False
    for line in lines:
        match = process_pattern.search(line)
        if not match:
            continue
        row_gpu_index = int(match.group(1))
        row_pid = int(match.group(2))
        row_memory_mb = float(match.group(3))
        if row_gpu_index != gpu_index:
            continue
        if process_pid is not None and row_pid != process_pid:
            continue
        process_total += row_memory_mb
        matched_process = True

    if matched_process:
        return process_total

    output = result.stdout.strip() or "<vazio>"
    if process_pid is None:
        _warn_nvidia_smi(f"nao encontrei Memory-Usage/processos da GPU {gpu_index} na saida: {output}")
    else:
        _warn_nvidia_smi(f"nao encontrei VRAM do PID {process_pid} na GPU {gpu_index} na saida: {output}")
    return None


class GpuMemorySampler:
    def __init__(self, gpu_index: int, process_pid: Optional[int], interval: float) -> None:
        self.gpu_index = gpu_index
        self.process_pid = process_pid
        self.interval = interval
        self.peak_mb: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self.peak_mb = _gpu_memory_used_mb(self.gpu_index, self.process_pid)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[float]:
        self._running = False
        if self._thread is not None:
            self._thread.join()
        current = _gpu_memory_used_mb(self.gpu_index, self.process_pid)
        if current is not None:
            self.peak_mb = max(self.peak_mb or current, current)
        return self.peak_mb

    def _sample(self) -> None:
        while self._running:
            current = _gpu_memory_used_mb(self.gpu_index, self.process_pid)
            if current is not None:
                self.peak_mb = max(self.peak_mb or current, current)
            sleep(self.interval)


def _format_mb(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f} MB"


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _nested_shape(value: Any) -> List[int]:
    shape = []
    while isinstance(value, list):
        shape.append(len(value))
        value = value[0] if value else []
    return shape


def _inspect_variant_response_shape(
    response: Dict[str, Any],
    requested_outputs: List[str],
    ontology_terms: List[str],
    strands: List[str],
) -> None:
    expected_tracks = len(ontology_terms) * len(strands)
    variant_outputs = response.get("variant_outputs", [])
    if not variant_outputs:
        print("Checagem de saida: resposta sem variant_outputs")
        return

    output_name = requested_outputs[0]
    first = variant_outputs[0]
    print("\n--- Checagem de saida ---")
    print(
        f"esperado se tracks = ontologias x strands: "
        f"{len(ontology_terms)} x {len(strands)} = {expected_tracks}"
    )
    for allele in ("reference", "alternate"):
        track = first.get(allele, {}).get(output_name)
        if track is None:
            print(f"{allele}.{output_name}: ausente")
            continue
        values = track.get("values", [])
        names = track.get("names", [])
        shape = _nested_shape(values)
        column_count = shape[-1] if len(shape) >= 2 else len(names)
        effective_columns = column_count * len(strands)
        print(
            f"{allele}.{output_name}: shape={shape} names={len(names)} "
            f"colunas_por_input={column_count} "
            f"colunas_por_individuo_com_strands_expandidos={effective_columns}"
        )
        if names:
            print(f"{allele}.{output_name}.names={names}")
            unique_names = list(dict.fromkeys(names))
            if len(unique_names) != len(names):
                print(
                    f"{allele}.{output_name}: {len(names) - len(unique_names)} "
                    "tracks duplicados pelo nome"
                )
            for term in ontology_terms:
                term_count = sum(1 for name in names if name.startswith(term))
                print(f"{allele}.{output_name}: {term} -> {term_count} tracks")
        if column_count == expected_tracks:
            print(f"{allele}.{output_name}: OK, tensor parece ser <dimensao grande>:{expected_tracks}")
        elif effective_columns == expected_tracks:
            print(
                f"{allele}.{output_name}: OK agregado por individuo: "
                f"{column_count} colunas/input x {len(strands)} strands = {expected_tracks}"
            )
        else:
            print(
                f"{allele}.{output_name}: ATENCAO, colunas nao batem com {expected_tracks}; "
                "ontologias filtram tracks, mas cada ontologia pode ter varios tracks RNA-seq"
            )


def _expand_inputs_by_strand(
    intervals: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    strands: List[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    expanded_intervals = []
    expanded_variants = []
    for interval, variant in zip(intervals, variants, strict=True):
        for strand in strands:
            interval_with_strand = dict(interval)
            interval_with_strand["strand"] = strand
            expanded_intervals.append(interval_with_strand)
            expanded_variants.append(dict(variant))
    return expanded_intervals, expanded_variants


def _result_row(
    mode: str,
    round_index: int,
    num_individuals: int,
    model_inputs: int,
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
        "model_inputs": model_inputs,
        "total_duration_s": f"{total_duration_s:.6f}",
        "duration_per_individual_s": f"{total_duration_s / num_individuals:.6f}",
        "duration_per_model_input_s": f"{total_duration_s / model_inputs:.6f}",
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
        "model_inputs",
        "total_duration_s",
        "duration_per_individual_s",
        "duration_per_model_input_s",
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
    model_inputs: int,
    gpu_index: int,
    gpu_process_pid: Optional[int],
    poll_interval: float,
    call: Callable[[], None],
) -> Dict[str, Any]:
    vram_before = _gpu_memory_used_mb(gpu_index, gpu_process_pid)
    sampler = GpuMemorySampler(gpu_index, gpu_process_pid, poll_interval)
    sampler.start()
    start = perf_counter()
    call()
    total_duration = perf_counter() - start
    vram_peak = sampler.stop()
    vram_after = _gpu_memory_used_mb(gpu_index, gpu_process_pid)

    row = _result_row(
        mode=mode,
        round_index=round_index,
        num_individuals=num_individuals,
        model_inputs=model_inputs,
        total_duration_s=total_duration,
        vram_before_mb=vram_before,
        vram_peak_mb=vram_peak,
        vram_after_mb=vram_after,
    )
    print(
        f"{mode:<10} rodada {round_index:>2}: "
        f"total={_format_seconds(total_duration)} "
        f"por_individuo={_format_seconds(total_duration / num_individuals)} "
        f"por_input={_format_seconds(total_duration / model_inputs)} "
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
    num_individuals: int,
    gpu_index: int,
    gpu_process_pid: Optional[int],
    poll_interval: float,
) -> List[Dict[str, Any]]:
    rows = []
    model_inputs = len(variants)

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
        try:
            rows.append(
                _measure_call(
                    mode="individual",
                    round_index=round_index,
                    num_individuals=num_individuals,
                    model_inputs=model_inputs,
                    gpu_index=gpu_index,
                    gpu_process_pid=gpu_process_pid,
                    poll_interval=poll_interval,
                    call=run_individual,
                )
            )
            rows.append(
                _measure_call(
                    mode="batch",
                    round_index=round_index,
                    num_individuals=num_individuals,
                    model_inputs=model_inputs,
                    gpu_index=gpu_index,
                    gpu_process_pid=gpu_process_pid,
                    poll_interval=poll_interval,
                    call=run_batch,
                )
            )
        except KeyboardInterrupt:
            print("\nBenchmark interrompido pelo usuario; retornando resultados completos ja coletados.")
            return rows
    return rows


def _float_values(rows: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(row[key]) for row in rows if row[key] != ""]


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    print("\nResumo")
    print(
        "| modo | rodadas | tempo total medio | tempo medio/individuo | "
        "tempo medio/input | tempo melhor/individuo | VRAM pico media | "
        "VRAM pico maxima | delta VRAM medio |"
    )
    print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ("individual", "batch"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        total_times = _float_values(mode_rows, "total_duration_s")
        per_individual_times = _float_values(mode_rows, "duration_per_individual_s")
        per_model_input_times = _float_values(mode_rows, "duration_per_model_input_s")
        vram_peaks = _float_values(mode_rows, "vram_peak_mb")
        vram_deltas = _float_values(mode_rows, "vram_peak_delta_mb")
        print(
            f"| {mode} | {len(mode_rows)} | {_format_seconds(mean(total_times))} | "
            f"{_format_seconds(mean(per_individual_times))} | "
            f"{_format_seconds(mean(per_model_input_times))} | "
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
    parser.add_argument(
        "--strands",
        default=DEFAULT_STRANDS,
        help="Strands por individuo, separados por virgula. Default: +,-.",
    )
    parser.add_argument("--base-url", default=_default_base_url())
    parser.add_argument("--gpu-index", type=int, default=0, help="Indice da GPU monitorada pelo nvidia-smi.")
    parser.add_argument(
        "--gpu-process-pid",
        type=int,
        help="Opcional: mede apenas a VRAM desse PID na tabela de processos do nvidia-smi.",
    )
    parser.add_argument("--window-size", type=int, default=_env_int("ALPHAGENOME_SEQUENCE_LEN", DEFAULT_SEQUENCE_LEN))
    parser.add_argument("--start", type=int, default=1_000_000)
    parser.add_argument("--ontology-terms", default=DEFAULT_ONTOLOGY_TERMS)
    parser.add_argument("--requested-outputs", default="RNA_SEQ")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--results-csv", default=os.getenv("ALPHAGENOME_INDIVIDUAL_RESULTS_CSV", DEFAULT_RESULTS_CSV))
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    client = AlphaGenomeClient(base_url=args.base_url)
    base_intervals, base_variants = build_variant_inputs(
        start=args.start,
        window_size=args.window_size,
        num_variants=args.num_individuals,
    )
    strands = _parse_csv_list(args.strands)
    intervals, variants = _expand_inputs_by_strand(base_intervals, base_variants, strands)
    ontology_terms = _parse_csv_list(args.ontology_terms)
    requested_outputs = _parse_csv_list(args.requested_outputs)

    print("--- Configuracao ---")
    print(f"base_url={args.base_url}")
    print(f"gpu_index={args.gpu_index}")
    print(f"gpu_process_pid={args.gpu_process_pid or 'todos'}")
    print(f"window_size={args.window_size}")
    print(f"num_individuals={args.num_individuals}")
    print(f"strands={','.join(strands)}")
    print(f"model_inputs={len(variants)}")
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
        warmup_batch_response = client.predict_variants_batch(
            intervals=intervals,
            variants=variants,
            ontology_terms=ontology_terms,
            requested_outputs=requested_outputs,
            batch_size=args.batch_size,
        )
        _inspect_variant_response_shape(
            response=warmup_batch_response,
            requested_outputs=requested_outputs,
            ontology_terms=ontology_terms,
            strands=strands,
        )
    except requests.exceptions.HTTPError as e:
        print(f"Erro HTTP no warmup: {e.response.status_code} - {e.response.text}")
        raise

    print("\n--- Benchmark: individual sequencial vs batch ---")
    rows = []
    try:
        rows = benchmark_individual_vs_batch(
            client=client,
            intervals=intervals,
            variants=variants,
            ontology_terms=ontology_terms,
            requested_outputs=requested_outputs,
            batch_size=args.batch_size,
            rounds=args.rounds,
            num_individuals=args.num_individuals,
            gpu_index=args.gpu_index,
            gpu_process_pid=args.gpu_process_pid,
            poll_interval=args.poll_interval,
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrompido pelo usuario; salvando resultados completos ja coletados.")

    if rows:
        _print_summary(rows)
    else:
        print("Nenhuma rodada completa coletada.")

    if rows and not args.no_csv:
        _write_csv(args.results_csv, rows)
        print(f"CSV salvo em: {args.results_csv}")


if __name__ == "__main__":
    main()
