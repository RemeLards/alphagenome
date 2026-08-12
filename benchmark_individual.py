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
DEFAULT_HAPLOTYPES = "H1,H2"
DEFAULT_STRANDS = "."
DEFAULT_ONTOLOGY_TERMS = "CL:1000458,CL:0000346,CL:2000092"
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
    haplotypes: List[str],
    strands: List[str],
) -> None:
    variant_outputs = response.get("variant_outputs", [])
    if not variant_outputs:
        print("Checagem de saida: resposta sem variant_outputs")
        return
    if len(variant_outputs) < len(strands):
        print(
            "Checagem de saida: resposta nao tem entradas suficientes para "
            f"{len(strands)} strands"
        )
        return

    output_name = requested_outputs[0]
    expected_tracks = len(ontology_terms) * len(strands)
    print("\n--- Checagem de saida ---")
    print(
        f"esperado pelo professor: {len(ontology_terms)} ontologias x "
        f"{len(haplotypes)} haplotipos = {len(ontology_terms) * len(haplotypes)} canais"
    )
    print(
        f"strands={','.join(strands)}; use +,- apenas se quiser testar orientacao genomica"
    )
    for allele in ("reference", "alternate"):
        selected_names = []
        position_count: Optional[int] = None
        missing = []
        for haplotype_index, haplotype in enumerate(haplotypes):
            track = variant_outputs[haplotype_index].get(allele, {}).get(output_name)
            if track is None:
                missing.append(f"{haplotype}:output ausente")
                continue
            shape = _nested_shape(track.get("values", []))
            names = track.get("names", [])
            if shape:
                position_count = shape[0]
            print(
                f"{allele}.{output_name} haplotype={haplotype}: "
                f"raw_shape={shape} raw_names={len(names)}"
            )
            for term in ontology_terms:
                matches = [name for name in names if name.startswith(term)]
                if not matches:
                    missing.append(f"{haplotype}:{term}")
                    continue
                selected_names.append(f"{haplotype}:{matches[0]}")
        logical_shape = [position_count or 0, len(selected_names)]
        print(f"{allele}.{output_name}: tensor_logico_shape={logical_shape}")
        print(f"{allele}.{output_name}: canais_selecionados={selected_names}")
        expected_haplotype_tracks = len(ontology_terms) * len(haplotypes)
        if len(selected_names) == expected_haplotype_tracks and not missing:
            print(
                f"{allele}.{output_name}: OK, equivale a "
                f"<dimensao grande>:{expected_haplotype_tracks} apos combinar haplotipos"
            )
        else:
            print(
                f"{allele}.{output_name}: ATENCAO, selecionei {len(selected_names)}/"
                f"{expected_haplotype_tracks} canais; faltando={missing}"
            )


def _expand_inputs(
    intervals: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    haplotypes: List[str],
    strands: List[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    expanded_intervals = []
    expanded_variants = []
    for interval, variant in zip(intervals, variants, strict=True):
        for _haplotype in haplotypes:
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
    haplotypes: List[str],
    strands: List[str],
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
        "haplotypes": ";".join(haplotypes),
        "strands": ";".join(strands),
        "total_duration_s": f"{total_duration_s:.6f}",
        "duration_per_individual_s": f"{total_duration_s / num_individuals:.6f}",
        "duration_per_model_input_s": f"{total_duration_s / model_inputs:.6f}",
        "vram_before_mb": "" if vram_before_mb is None else f"{vram_before_mb:.3f}",
        "vram_peak_mb": "" if vram_peak_mb is None else f"{vram_peak_mb:.3f}",
        "vram_after_mb": "" if vram_after_mb is None else f"{vram_after_mb:.3f}",
        "vram_peak_delta_mb": "" if peak_delta is None else f"{peak_delta:.3f}",
        "vram_peak_delta_per_individual_mb": ""
        if peak_delta is None
        else f"{peak_delta / num_individuals:.3f}",
        "vram_peak_delta_per_model_input_mb": ""
        if peak_delta is None
        else f"{peak_delta / model_inputs:.3f}",
    }


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "round",
        "num_individuals",
        "model_inputs",
        "haplotypes",
        "strands",
        "total_duration_s",
        "duration_per_individual_s",
        "duration_per_model_input_s",
        "vram_before_mb",
        "vram_peak_mb",
        "vram_after_mb",
        "vram_peak_delta_mb",
        "vram_peak_delta_per_individual_mb",
        "vram_peak_delta_per_model_input_mb",
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
    haplotypes: List[str],
    strands: List[str],
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
        haplotypes=haplotypes,
        strands=strands,
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
        f"delta_pico={_format_mb(float(row['vram_peak_delta_mb']) if row['vram_peak_delta_mb'] else None)} "
        f"delta_por_individuo={_format_mb(float(row['vram_peak_delta_per_individual_mb']) if row['vram_peak_delta_per_individual_mb'] else None)}"
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
    haplotypes: List[str],
    strands: List[str],
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
                    haplotypes=haplotypes,
                    strands=strands,
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
                    haplotypes=haplotypes,
                    strands=strands,
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
        "VRAM pico maxima | delta VRAM medio | delta VRAM/individuo | delta VRAM/input |"
    )
    print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ("individual", "batch"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        total_times = _float_values(mode_rows, "total_duration_s")
        per_individual_times = _float_values(mode_rows, "duration_per_individual_s")
        per_model_input_times = _float_values(mode_rows, "duration_per_model_input_s")
        vram_peaks = _float_values(mode_rows, "vram_peak_mb")
        vram_deltas = _float_values(mode_rows, "vram_peak_delta_mb")
        vram_deltas_per_individual = _float_values(mode_rows, "vram_peak_delta_per_individual_mb")
        vram_deltas_per_input = _float_values(mode_rows, "vram_peak_delta_per_model_input_mb")
        print(
            f"| {mode} | {len(mode_rows)} | {_format_seconds(mean(total_times))} | "
            f"{_format_seconds(mean(per_individual_times))} | "
            f"{_format_seconds(mean(per_model_input_times))} | "
            f"{_format_seconds(min(per_individual_times))} | "
            f"{_format_mb(mean(vram_peaks) if vram_peaks else None)} | "
            f"{_format_mb(max(vram_peaks) if vram_peaks else None)} | "
            f"{_format_mb(mean(vram_deltas) if vram_deltas else None)} | "
            f"{_format_mb(mean(vram_deltas_per_individual) if vram_deltas_per_individual else None)} | "
            f"{_format_mb(mean(vram_deltas_per_input) if vram_deltas_per_input else None)} |"
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
        description="Compara tempo e VRAM usando rna_seq e as ontologias do genes_1000_all_snps_only.yaml."
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--num-individuals", type=int, default=DEFAULT_NUM_INDIVIDUALS)
    parser.add_argument("--batch-size", type=int, default=_env_int("ALPHAGENOME_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    parser.add_argument(
        "--haplotypes",
        default=DEFAULT_HAPLOTYPES,
        help="Haplotipos por individuo, separados por virgula. Default conforme genes_1000_all_snps_only.yaml: H1,H2.",
    )
    parser.add_argument(
        "--strands",
        default=DEFAULT_STRANDS,
        help="Orientacoes genomicas por haplotipo. Default: .; use +,- apenas se quiser testar orientacao.",
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
    haplotypes = _parse_csv_list(args.haplotypes)
    strands = _parse_csv_list(args.strands)
    intervals, variants = _expand_inputs(base_intervals, base_variants, haplotypes, strands)
    ontology_terms = _parse_csv_list(args.ontology_terms)
    requested_outputs = _parse_csv_list(args.requested_outputs)

    print("--- Configuracao ---")
    print(f"base_url={args.base_url}")
    print(f"gpu_index={args.gpu_index}")
    print(f"gpu_process_pid={args.gpu_process_pid or 'todos'}")
    print(f"window_size={args.window_size}")
    print(f"num_individuals={args.num_individuals}")
    print(f"haplotypes={','.join(haplotypes)}")
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
            haplotypes=haplotypes,
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
            haplotypes=haplotypes,
            strands=strands,
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
