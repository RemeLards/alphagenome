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
hf_login(settings.hf_token)

class AlphaGenomeEngine:
    def __init__(
        self,
        model_name: str = "all_folds",
        input_size: int = SequenceLength.KB_8,
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

        ref_dict: dict[str, TrackDataSchema | None] = {}
        alt_dict: dict[str, TrackDataSchema | None] = {}
        for ot in out_types:
            name = ot.name
            ref_dict[name] = self._track_to_schema(
                getattr(result.reference, name.lower(), None)
            )
            alt_dict[name] = self._track_to_schema(
                getattr(result.alternate, name.lower(), None)
            )

        return VariantOutputSchema(reference=ref_dict, alternate=alt_dict)

    def predict_variants_batch(
        self,
        intervals: list[IntervalSchema],
        variants: list[VariantSchema],
        ontology_terms: list[str] | None = None,
        requested_outputs: list[str] | None = None,
    ) -> list[VariantOutputSchema]:
        gintervals = [self._build_interval(i) for i in intervals]
        gvariants = [self._build_variant(v) for v in variants]
        terms = ontology_terms or ["UBERON:0001157"]
        outputs = requested_outputs or ["RNA_SEQ"]
        out_types = [self._parse_output_type(o) for o in outputs]

        results = self._model.predict_variants(
            intervals=gintervals,
            variants=gvariants,
            ontology_terms=terms,
            requested_outputs=out_types,
        )

        parsed: list[VariantOutputSchema] = []
        for result in results:
            ref_dict: dict[str, TrackDataSchema | None] = {}
            alt_dict: dict[str, TrackDataSchema | None] = {}
            for ot in out_types:
                name = ot.name
                ref_dict[name] = self._track_to_schema(
                    getattr(result.reference, name.lower(), None)
                )
                alt_dict[name] = self._track_to_schema(
                    getattr(result.alternate, name.lower(), None)
                )
            parsed.append(VariantOutputSchema(reference=ref_dict, alternate=alt_dict))
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
    settings.model_name,
    settings.sequence_len,
)
