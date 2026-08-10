from src.config import (
    settings,
    SequenceLength,
)
from alphagenome.data import genome
from alphagenome_research.model import dna_model
from src.schemas import (
    IntervalSchema,
    VariantSchema,
    TrackDataSchema,
    VariantOutputSchema,
)
from huggingface_hub import login as hf_login
from time import time
import numpy as np

hf_login(settings.HF_TOKEN)

class AlphaGenomeEngine:
    def __init__(
        self,
        model_name: str = "all_folds",
        input_size: int = SequenceLength.K_8,
    ):
        self._model_name = model_name
        self._input_size = input_size

        self._model = dna_model.create_from_huggingface(model_name)

        self._warmup()
    
    def _warmup(self):
        """Forca a compilacao JAX (JIT) tanto para interval quanto para variant."""
        print("🔥 [Warmup] Compilando grafos JAX do AlphaGenome...")
        start_time = time()

        # Intervalo dummy dentro dos limites validos
        interval = genome.Interval(
            chromosome="chr1",
            start=1_000_000,
            end=1_000_000 + self._input_size,
            strand="+",
        )

        # Variante dummy estritamente dentro do intervalo
        variant = genome.Variant(
            chromosome="chr1",
            position=1_000_500,
            reference_bases="A",
            alternate_bases="G",
        )

        # Usar as saidas mais comuns para aquecer os caminhos do modelo
        warmup_outputs = [
            dna_model.OutputType.RNA_SEQ,
            dna_model.OutputType.ATAC,
        ]

        try:
            # 1. Compila o grafo de predicao de intervalo
            self._model.predict_interval(
                interval=interval,
                ontology_terms=["UBERON:0001157"],
                requested_outputs=warmup_outputs,
            )

            # 2. Compila o grafo de predicao de variante (compara ref vs alt)
            self._model.predict_variant(
                interval=interval,
                variant=variant,
                ontology_terms=["UBERON:0001157"],
                requested_outputs=warmup_outputs,
            )
            print(f"✅ [Warmup] Concluido com sucesso em {time() - start_time:.2f}s")
        except Exception as e:
            print(f"⚠️ [Warmup] Falha no Warmup do modelo: {e}")
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── internal builders ──────────────────────────────────────────────

    @staticmethod
    def _build_interval(s: IntervalSchema) -> genome.Interval:
        return genome.Interval(
            chromosome=s.chromosome,
            start=s.start,
            end=s.end,
            strand=s.strand,
        )

    @staticmethod
    def _build_variant(s: VariantSchema) -> genome.Variant:
        return genome.Variant(
            chromosome=s.chromosome,
            position=s.position,
            reference_bases=s.reference_bases,
            alternate_bases=s.alternate_bases,
        )

    @staticmethod
    def _parse_output_type(name: str) -> dna_model.OutputType:
        return dna_model.OutputType[name.upper()]

    @staticmethod
    def _track_to_schema(track) -> TrackDataSchema | None:
        if track is None:
            return None
        return TrackDataSchema(
            values=track.values.tolist(),
            names=list(track.names),
            resolution=track.resolution,
            interval=IntervalSchema(
                chromosome=track.interval.chromosome,
                start=track.interval.start,
                end=track.interval.end,
                strand=track.interval.strand,
            )
            if track.interval
            else None,
        )

    @classmethod
    def _variant_output_to_schema(
        cls,
        result,
        out_types: list[dna_model.OutputType],
    ) -> VariantOutputSchema:
        ref_dict: dict[str, TrackDataSchema | None] = {}
        alt_dict: dict[str, TrackDataSchema | None] = {}
        for ot in out_types:
            name = ot.name
            ref_dict[name] = cls._track_to_schema(
                getattr(result.reference, name.lower(), None)
            )
            alt_dict[name] = cls._track_to_schema(
                getattr(result.alternate, name.lower(), None)
            )
        return VariantOutputSchema(reference=ref_dict, alternate=alt_dict)

    # ── public API ─────────────────────────────────────────────────────

    def predict_variant(
        self,
        interval: IntervalSchema,
        variant: VariantSchema,
        ontology_terms: list[str] | None = None,
        requested_outputs: list[str] | None = None,
    ) -> VariantOutputSchema:
        ginterval = self._build_interval(interval)
        gvariant = self._build_variant(variant)
        terms = ontology_terms or ["UBERON:0001157"]
        outputs = requested_outputs or ["RNA_SEQ"]
        out_types = [self._parse_output_type(o) for o in outputs]

        result = self._model.predict_variant(
            interval=ginterval,
            variant=gvariant,
            ontology_terms=terms,
            requested_outputs=out_types,
        )

        return self._variant_output_to_schema(result, out_types)

    def _predict_variants_chunk_vectorized(
        self,
        intervals: list[genome.Interval],
        variants: list[genome.Variant],
        ontology_terms: list[str],
        out_types: list[dna_model.OutputType],
    ) -> list[VariantOutputSchema]:
        # O predict_variants publico do AlphaGenome usa ThreadPoolExecutor, mas o
        # modelo de research serializa a entrada na GPU com o lock de
        # _device_context. Aqui chamamos o _predict_variant jittado uma vez com
        # batch B > 1, seguindo o modelo normal de arrays batched do JAX:
        # https://docs.jax.dev/en/latest/jax-101/03-vectorization.html
        # Os simbolos internos abaixo vem de alphagenome_research.model.dna_model.
        organism = dna_model.Organism.HOMO_SAPIENS
        requested_outputs = tuple(set(out_types))
        converted_terms = dna_model._convert_ontology_terms(ontology_terms)
        metadata = self._model._metadata[organism]
        track_masks = dna_model.metadata_lib.create_track_masks(
            metadata,
            requested_outputs=requested_outputs,
            requested_ontologies=converted_terms,
        )

        fasta_extractor = self._model._get_fasta_extractor(organism)
        reference_sequences: list[np.ndarray] = []
        alternate_sequences: list[np.ndarray] = []
        reference_gene_masks: list[np.ndarray] = []
        splice_sites: list[np.ndarray] | None = []
        indel_masks = []

        gene_mask_extractor = self._model._gene_mask_extractors.get(organism)
        splice_site_extractor = self._model._splice_site_extractors.get(organism)

        for interval, variant in zip(intervals, variants, strict=True):
            reference_sequence, alternate_sequence = (
                dna_model.genome_io.extract_variant_sequences(
                    interval, variant, fasta_extractor
                )
            )
            reference_sequences.append(
                np.asarray(self._model._one_hot_encoder.encode(reference_sequence))
            )
            alternate_sequences.append(
                np.asarray(self._model._one_hot_encoder.encode(alternate_sequence))
            )

            reference_gene_mask = np.ones((interval.width, 1), dtype=bool)
            if gene_mask_extractor:
                mask, _ = gene_mask_extractor.extract(interval, variant)
                if mask.size > 0:
                    reference_gene_mask = mask.max(-1, keepdims=True)
            reference_gene_masks.append(reference_gene_mask)

            if splice_site_extractor:
                splice_sites.append(
                    splice_site_extractor.extract(interval) * reference_gene_mask
                )
            else:
                splice_sites = None

            indel_masks.append(
                dna_model.variant_scoring.IndelMask.from_variant(variant, interval)
            )

        reference_sequence_batch = np.stack(reference_sequences)
        alternate_sequence_batch = np.stack(alternate_sequences)
        reference_gene_mask_batch = np.stack(reference_gene_masks)
        splice_site_batch = np.stack(splice_sites) if splice_sites is not None else None
        # IndelMask e um pytree; este padrao transforma uma lista de IndelMask
        # com a mesma estrutura em um unico IndelMask batched, empilhando cada
        # folha correspondente. Docs: https://docs.jax.dev/en/latest/pytrees.html
        # API: https://docs.jax.dev/en/latest/_autosummary/jax.tree.map.html
        indel_mask_batch = dna_model.jax.tree.map(
            lambda *xs: np.stack(xs),
            *indel_masks,
        )
        splice_junction_masks = dna_model._SpliceJunctionVariantMasks(
            splice_sites=splice_site_batch,
            reference_genes=reference_gene_mask_batch,
            indel_masks=indel_mask_batch,
        )

        with self._model._device_context as device, dna_model.jax.transfer_guard(
            "disallow"
        ):
            # transfer_guard("disallow") bloqueia transferencias implicitas, mas
            # permite device_put/device_get explicitos. Assim fica visivel se uma
            # conversao Python/NumPy -> device escapou sem querer.
            # https://docs.jax.dev/en/latest/transfer_guard.html
            # device_put move explicitamente o batch completo [B, S, 4] e os
            # pytrees auxiliares para o device escolhido antes da chamada jittada.
            # A transferencia e assincrona e o resultado fica committed ao device.
            # https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html
            reference_predictions, alternate_predictions = self._model._predict_variant(
                self._model._params,
                self._model._state,
                dna_model.jax.device_put(reference_sequence_batch, device),
                dna_model.jax.device_put(alternate_sequence_batch, device),
                dna_model.jax.device_put(splice_junction_masks, device),
                dna_model.jax.device_put(
                    np.full(
                        (len(variants),),
                        dna_model.convert_to_organism_index(organism),
                        dtype=np.int32,
                    ),
                    device,
                ),
                requested_outputs=requested_outputs,
                negative_strand_mask=dna_model.jax.device_put(
                    np.asarray([interval.negative_strand for interval in intervals]),
                    device,
                ),
                strand_reindexing=dna_model.jax.device_put(
                    metadata.strand_reindexing,
                    device,
                ),
                indel_stitch_input=None,
            )
            reference_predictions, alternate_predictions = (
                dna_model._filter_variant_predictions(
                    reference_predictions,
                    alternate_predictions,
                    track_masks=dna_model.jax.device_put(track_masks, device),
                )
            )
        # device_get transfere a arvore de predicoes para o host; se x e um
        # pytree, os buffers sao copiados em paralelo. Fazemos isso uma vez para
        # o batch inteiro e depois fatiamos no NumPy/Python, evitando muitas
        # leituras pequenas por output/item. Relacionado ao dispatch assincrono:
        # https://docs.jax.dev/en/latest/_autosummary/jax.device_get.html
        # https://docs.jax.dev/en/latest/async_dispatch.html
        reference_predictions = dna_model.jax.device_get(reference_predictions)
        alternate_predictions = dna_model.jax.device_get(alternate_predictions)

        parsed: list[VariantOutputSchema] = []
        for index, interval in enumerate(intervals):
            # Depois do device_get, cada folha da arvore tem eixo de batch; este
            # tree.map extrai o item index de todas as folhas preservando a
            # estrutura esperada por _construct_output_from_predictions.
            reference_prediction = dna_model.jax.tree.map(
                lambda x: x[index], reference_predictions
            )
            alternate_prediction = dna_model.jax.tree.map(
                lambda x: x[index], alternate_predictions
            )
            result = dna_model.VariantOutput(
                reference=dna_model._construct_output_from_predictions(
                    reference_prediction,
                    track_masks=track_masks,
                    metadata=metadata,
                    interval=interval,
                ),
                alternate=dna_model._construct_output_from_predictions(
                    alternate_prediction,
                    track_masks=track_masks,
                    metadata=metadata,
                    interval=interval,
                ),
            )
            parsed.append(self._variant_output_to_schema(result, out_types))
        return parsed

    def predict_variants_batch(
        self,
        intervals: list[IntervalSchema],
        variants: list[VariantSchema],
        ontology_terms: list[str] | None = None,
        requested_outputs: list[str] | None = None,
        max_workers: int | None = None,
        batch_size: int | None = None,
    ) -> list[VariantOutputSchema]:
        gintervals = [self._build_interval(i) for i in intervals]
        gvariants = [self._build_variant(v) for v in variants]
        terms = ontology_terms or ["UBERON:0001157"]
        outputs = requested_outputs or ["RNA_SEQ"]
        out_types = [self._parse_output_type(o) for o in outputs]

        parsed: list[VariantOutputSchema] = []
        chunk_size = batch_size or max_workers or settings.BATCH_SIZE
        for start in range(0, len(gvariants), chunk_size):
            parsed.extend(
                self._predict_variants_chunk_vectorized(
                    intervals=gintervals[start : start + chunk_size],
                    variants=gvariants[start : start + chunk_size],
                    ontology_terms=terms,
                    out_types=out_types,
                )
            )
        return parsed

    def predict_interval(
        self,
        interval: IntervalSchema,
        ontology_terms: list[str] | None = None,
        requested_outputs: list[str] | None = None,
    ) -> dict[str, TrackDataSchema | None]:
        ginterval = self._build_interval(interval)
        terms = ontology_terms or ["UBERON:0001157"]
        outputs = requested_outputs or ["RNA_SEQ"]
        out_types = [self._parse_output_type(o) for o in outputs]

        result = self._model.predict_interval(
            interval=ginterval,
            ontology_terms=terms,
            requested_outputs=out_types,
        )

        out_dict: dict[str, TrackDataSchema | None] = {}
        for ot in out_types:
            name = ot.name
            out_dict[name] = self._track_to_schema(
                getattr(result, name.lower(), None)
            )
        return out_dict


alphagenome_model = AlphaGenomeEngine(
    settings.MODEL_NAME,
    settings.SEQUENCE_LEN,
)
