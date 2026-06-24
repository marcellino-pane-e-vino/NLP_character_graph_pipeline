# Relationship Extraction Module — Architectural and Implementation Specification

This document describes the proposed relation-extraction module for the NLP + ontology-population pipeline. It is designed to fit the existing `pipeline.ipynb` organization:

```text
# Setup
# Imports
# Functions
# Config
## I/O config
## Runtime and pipeline config
# Main
## Node extraction
### Tokenization
### Chunking
### Coreference clusters extraction
### Cluster typing
### OCEAN scoring
## Edge extraction
```

The module intentionally follows the same design philosophy as `coref_schema.py`: primary data live in small dataclasses; secondary data are reached through indexes and helper methods.

---

## 1. High-level pipeline

```text
doc._.coref_layer
        |
        v
extract_relation_candidates.py
        |
        v
routed_relation_candidates.jsonl
        |
        v
align_relation_assignments.py
        |
        v
relation_assignments.csv
        |
        v
aggregate_cluster_assertions.py
        |
        v
cluster_assertions.csv
        |
        v
annotate_relation_layer.py
        |
        v
doc._.relation_layer
        |
        v
ontology_managment.py / ABox population
```

The core idea is:

```text
RelationMention     = textual anchor: source mention + predicate trigger + target mention
RelationAssignment  = neural object-property scoring for one RelationMention
ClusterAssertion    = aggregated cluster-level assertion for KG population
RelationLayer       = single Doc-level relation wrapper containing assignments, assertions, indexes, and helpers
```

There is no `relation_mention_layer` and no `relation_assertion_layer`. There is only:

```python
doc._.relation_layer
```

---

## 2. Files and responsibilities

### 2.1 `relationship_extraction/relation_schema.py`

Defines the final relation schema stored in the Doc.

Contains:

```python
RelationMention
RelationAssignment
ClusterAssertion
RelationLayer
softmax_dict
make_relation_mention_id
make_cluster_assertion_id
register_spacy_relation_extension
require_relation_layer
```

Does not:

- extract relations;
- call the neural model;
- parse CSV/JSONL;
- mutate the ontology;
- duplicate coreference data.

It imports:

```python
from coreference.coref_schema import Mention, Cluster, CorefLayer, require_coref_layer
```

#### `RelationMention`

Primary data only:

```python
@dataclass(frozen=True, slots=True)
class RelationMention:
    relation_mention_id: str
    source_mention: Mention
    predicate_token_i: int
    predicate_start: int
    predicate_end: int
    target_mention: Mention
```

It means:

```text
this source mention is textually connected to this target mention by this predicate trigger
```

It deliberately does not store:

- canonical names;
- cluster ids as separate fields;
- class labels;
- predicate surface text;
- chosen object property;
- confidence/margin/entropy.

Those are either derivable from `Mention`, `CorefLayer`, `Doc`, or belong to `RelationAssignment`.

#### `RelationAssignment`

Primary data:

```python
@dataclass(frozen=True, slots=True)
class RelationAssignment:
    relation_mention: RelationMention
    object_property_logits: dict[str, float]
    selection_method: str
```

It means:

```text
for this RelationMention, these OWL object properties received these neural logits
```

It stores logits, not softmax, because softmax/chosen/confidence/margin/entropy are derived.

#### `ClusterAssertion`

Primary data:

```python
@dataclass(frozen=True, slots=True)
class ClusterAssertion:
    cluster_assertion_id: str
    source_cluster_id: int
    object_property_iri: str
    target_cluster_id: int
    support_assignment_ids: tuple[str, ...]
    aggregation_method: str
```

It means:

```text
in the final KG, source cluster -- object_property_iri --> target cluster
```

`object_property_iri` is the OWL object property IRI, not the textual predicate.

#### `RelationLayer`

Primary containers:

```python
assignments: dict[str, RelationAssignment]
cluster_assertions: dict[str, ClusterAssertion]
```

Derived indexes:

```python
predicate_token_to_assignment_ids: dict[int, list[str]]
source_cluster_to_assignment_ids: dict[int, list[str]]
target_cluster_to_assignment_ids: dict[int, list[str]]
source_cluster_to_assertion_ids: dict[int, list[str]]
target_cluster_to_assertion_ids: dict[int, list[str]]
object_property_to_assertion_ids: dict[str, list[str]]
```

Important helpers:

```python
source_mention(assignment_id)
target_mention(assignment_id)
source_cluster(coref, assignment_id)
target_cluster(coref, assignment_id)
source_span(doc, assignment_id)
target_span(doc, assignment_id)
predicate_span(doc, assignment_id)
evidence_span(doc, assignment_id)
object_property_scores(assignment_id)
chosen_object_property_iri(assignment_id)
confidence(assignment_id)
margin(assignment_id)
entropy(assignment_id)
assertion_assignments(assertion_id)
support_count(assertion_id)
mean_assertion_confidence(assertion_id)
summary()
```

---

### 2.2 `ontology/ontology_managment.py`

Extends the existing ontology helper file.

Existing responsibilities preserved:

- load ontology with OWLAPY;
- save ontology;
- validate object-property descriptions;
- build class DAG;
- assert cluster types;
- assert cluster object-property relations;
- assert cluster data-property values.

New relation-specific structures/functions:

```python
ObjectPropertySpec
build_relation_catalog
RelationRouter
build_relation_router
assert_cluster_object_property
```

#### `ObjectPropertySpec`

```python
@dataclass(frozen=True, slots=True)
class ObjectPropertySpec:
    iri: str
    local_name: str
    label: str
    human_readable_label: str
    description: str
    domains: tuple[str, ...]
    ranges: tuple[str, ...]
```

This is the compact object-property view used by stage 1 and stage 2.

#### `build_relation_catalog(...)`

Builds:

```python
dict[str, ObjectPropertySpec]
```

from the loaded OWL ontology.

The catalog includes label, human label, description, named class domains, and named class ranges.

V1 supports only named OWLClass domain/range routing. Complex class expressions are ignored in hard routing for now.

#### `RelationRouter`

Routes a source-target ontology type pair to candidate object properties:

```python
relation_router.candidates_for(source_class_iri, target_class_iri)
```

A property is accepted if:

```text
source_class_iri is the same as or subclass of one of the property domains
and
target_class_iri is the same as or subclass of one of the property ranges
```

The class DAG uses `parent -> child` edges.

---

### 2.3 `relationship_extraction/extract_relation_candidates.py`

Stage 1.

Input:

```python
doc
cluster_type_layer
relation_router
```

Output:

```text
routed_relation_candidates.jsonl
```

This file is based on the dependency-extraction code from the notebook.

It keeps the same dependency sets:

```python
SUBJECT_DEPS
PASSIVE_SUBJECT_DEPS
OBJECT_DEPS
PREP_DEPS
PREP_OBJECT_DEPS
AUX_PASSIVE_DEPS
NEGATION_DEPS
PARTICLE_DEPS
```

It also keeps the same core logic:

- require DEP and POS annotations;
- detect verbal predicates;
- collect subject dependents;
- collect direct objects;
- collect prepositional objects;
- resolve subject/object tokens through `CorefLayer`;
- prefer head-token mention matches;
- fall back to covering mentions;
- skip same-cluster source/target candidates;
- retain passive/negation flags;
- build normalized predicate labels such as `go_to`, `look_at`, `pick_up`.

Important functions:

```python
iter_syntax_relation_candidates(doc, cluster_type_layer=None)
extract_relation_candidates(doc, cluster_type_layer=None)
export_routed_relation_candidates_jsonl(doc, relation_router, output_path, cluster_type_layer=None, ...)
```

#### `iter_syntax_relation_candidates(...)`

Yields `SyntaxRelationCandidate` objects before ontology routing.

#### `extract_relation_candidates(...)`

Compatibility helper returning a `pandas.DataFrame` similar to the original notebook block.

#### `export_routed_relation_candidates_jsonl(...)`

Performs type-pair routing and writes JSONL rows. It does not save discarded rows. Discards are printed as requested:

```text
[subject][predicate][object] discarded because [reason]
```

Example:

```text
[Dorothy][go_to][Oz] discarded because [no_type_pair_candidates]
```

Output row shape:

```json
{
  "relation_mention_id": "pred_128_srcm_8_tgtm_31",
  "source_mention_id": 8,
  "predicate_token_i": 128,
  "predicate_start": 128,
  "predicate_end": 130,
  "target_mention_id": 31,
  "source_cluster_id": 12,
  "source_canonical_name": "Dorothy",
  "source_class_iri": "...#Character",
  "predicate": "go_to",
  "predicate_surface": "went to",
  "target_cluster_id": 45,
  "target_canonical_name": "Oz",
  "target_class_iri": "...#Place",
  "sentence_index": 42,
  "sentence_start": 1000,
  "sentence_end": 1014,
  "sentence_text": "Dorothy went to Oz.",
  "premise_text": "Context: Dorothy went to Oz.\n\nExtracted relation:\nDorothy -- went to -- Oz.",
  "candidate_properties": [
    {
      "iri": "...#travelsTo",
      "label": "travels to",
      "human_readable_label": "Travels To",
      "description": "...",
      "domains": ["...#Character"],
      "ranges": ["...#Place"]
    }
  ],
  "source_token_i": 1002,
  "target_token_i": 1005,
  "object_dependency": "pobj",
  "preposition": "to",
  "is_passive": false,
  "is_negated": false
}
```

---

### 2.4 `relationship_extraction/align_relation_assignments.py`

Stage 2.

Input:

```text
routed_relation_candidates.jsonl
```

Output:

```text
relation_assignments.csv
```

This stage does not know about OWLAPY, the router, or the Doc. It only scores the candidate properties already serialized by stage 1.

Important structures/functions:

```python
RelationNLIConfig
RelationPairScorer
TransformersRelationNLISelector
load_relation_nli_selector
relation_hypothesis_text
export_relation_assignments_csv
```

#### Neural scoring policy

For each routed candidate row:

```text
premise = premise_text from stage 1
hypothesis = template(label, description) for each candidate property
```

Then the model scores each pair:

```text
(relation_candidate, object_property)
```

The primary output is logits:

```python
object_property_logits = {
    "...#travelsTo": 4.7,
    "...#livesIn": 0.3,
}
```

If there is only one routed candidate property, no model call is needed. The output logit is `0.0`; softmax over one property gives `1.0`.

Output CSV columns:

```text
relation_mention_id
source_mention_id
predicate_token_i
predicate_start
predicate_end
target_mention_id
source_cluster_id
target_cluster_id
object_property_logits_json
selection_method
```

---

### 2.5 `relationship_extraction/aggregate_cluster_assertions.py`

Stage 3.

Input:

```text
relation_assignments.csv
```

Output:

```text
cluster_assertions.csv
```

Important structures/functions:

```python
RelationAggregationConfig
export_cluster_assertions_csv
```

V1 aggregation policy:

```text
group by (source_cluster_id, target_cluster_id)
for each assignment:
    softmax(object_property_logits)
    add scores to property accumulator
choose object property with maximum accumulated score
emit one ClusterAssertion for the cluster pair
```

Output CSV columns:

```text
cluster_assertion_id
source_cluster_id
object_property_iri
target_cluster_id
support_assignment_ids_json
aggregation_method
aggregated_score
```

`aggregated_score` is included in the CSV for debugging, but it is not a primary field in the final `ClusterAssertion` dataclass.

---

### 2.6 `relationship_extraction/annotate_relation_layer.py`

Final annotator.

Input:

```text
doc
relation_assignments.csv
cluster_assertions.csv
doc._.coref_layer
```

Output:

```python
doc._.relation_layer
```

Important functions:

```python
build_relation_layer_from_files
annotate_relation_layer_from_files
```

This module reconstructs rich Python objects:

```text
CSV source_mention_id -> coref.mentions[source_mention_id] -> Mention object
CSV target_mention_id -> coref.mentions[target_mention_id] -> Mention object
CSV logits JSON -> RelationAssignment
CSV assertion rows -> ClusterAssertion
```

It then creates:

```python
RelationLayer.from_data(assignments=..., cluster_assertions=...)
```

and attaches it to:

```python
doc._.relation_layer
```

---

## 3. Pipeline notebook integration

The notebook should remain an orchestrator. It should contain paths, config, calls, and sanity checks, not implementation logic.

### 3.1 Imports section

Add after ontology imports:

```python
from ontology.ontology_managment import (
    build_relation_catalog,
    build_relation_router,
    assert_cluster_object_property,
)

from relationship_extraction.extract_relation_candidates import (
    extract_relation_candidates,
    export_routed_relation_candidates_jsonl,
)

from relationship_extraction.align_relation_assignments import (
    RelationNLIConfig,
    load_relation_nli_selector,
    export_relation_assignments_csv,
)

from relationship_extraction.aggregate_cluster_assertions import (
    RelationAggregationConfig,
    export_cluster_assertions_csv,
)

from relationship_extraction.annotate_relation_layer import (
    annotate_relation_layer_from_files,
)

from relationship_extraction.relation_schema import (
    register_spacy_relation_extension,
    require_relation_layer,
)
```

### 3.2 `load_doc(...)` function

Add relation extension registration:

```python
register_spacy_relation_extension()
```

so the function becomes conceptually:

```python
def load_doc(path):
    ensure_booknlp_extensions()
    register_spacy_ontology_extension()
    register_spacy_ocean_extension()
    register_spacy_relation_extension()
    ...
```

### 3.3 I/O config section

Add:

```python
RELATION_OUTPUT_DIR = OUTPUT_ROOT / "relations"
RELATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTED_RELATION_CANDIDATES_PATH = RELATION_OUTPUT_DIR / "routed_relation_candidates.jsonl"
RELATION_ASSIGNMENTS_PATH = RELATION_OUTPUT_DIR / "relation_assignments.csv"
CLUSTER_ASSERTIONS_PATH = RELATION_OUTPUT_DIR / "cluster_assertions.csv"
RELATION_DOC_PATH = OUTPUT_ROOT / "relation_doc.pkl"
POPULATED_ONTOLOGY_PATH = OUTPUT_ROOT / "populated_ontology.ttl"
```

### 3.4 Runtime config section

Add:

```python
RELATION_PAIR_BATCH_SIZE = 64
RELATION_OVERWRITE_STAGE_1 = False
RELATION_OVERWRITE_STAGE_2 = False
RELATION_RESUME_STAGE_2 = True
RELATION_OVERWRITE_STAGE_3 = True
```

### 3.5 Ontology building section

After:

```python
onto, graph = load_tbox(ONTOLOGY_TTL_PATH, require_property_descriptions=True)
```

add:

```python
relation_catalog = build_relation_catalog(
    onto=onto,
    ontology_path=ONTOLOGY_TTL_PATH,
)

relation_router = build_relation_router(
    class_graph=graph,
    relation_catalog=relation_catalog,
)

print("Object properties:", len(relation_catalog))
```

### 3.6 Edge extraction section

Replace the notebook-local prototype code with calls to the module.

#### Optional preview

```python
relation_candidates_df = extract_relation_candidates(
    doc,
    cluster_type_layer=doc._.ontology_layer,
)

relation_candidates_df[
    [
        "source_cluster_id",
        "source_canonical_name",
        "source_class_iri",
        "predicate",
        "target_cluster_id",
        "target_canonical_name",
        "target_class_iri",
        "sentence_text",
        "is_passive",
        "is_negated",
    ]
]
```

#### Stage 1

```python
export_routed_relation_candidates_jsonl(
    doc=doc,
    cluster_type_layer=doc._.ontology_layer,
    relation_router=relation_router,
    output_path=ROUTED_RELATION_CANDIDATES_PATH,
    print_discards=True,
    overwrite=RELATION_OVERWRITE_STAGE_1,
)
```

#### Stage 2

```python
selector = load_relation_nli_selector(
    RelationNLIConfig(
        pair_batch_size=RELATION_PAIR_BATCH_SIZE,
    )
)

export_relation_assignments_csv(
    input_path=ROUTED_RELATION_CANDIDATES_PATH,
    output_path=RELATION_ASSIGNMENTS_PATH,
    selector=selector,
    overwrite=RELATION_OVERWRITE_STAGE_2,
    resume=RELATION_RESUME_STAGE_2,
)
```

#### Stage 3

```python
export_cluster_assertions_csv(
    assignments_path=RELATION_ASSIGNMENTS_PATH,
    output_path=CLUSTER_ASSERTIONS_PATH,
    aggregation_config=RelationAggregationConfig(
        aggregation_method="sum_softmax_by_cluster_pair",
        min_support_count=1,
        min_score=0.0,
    ),
    overwrite=RELATION_OVERWRITE_STAGE_3,
)
```

#### Final annotation

```python
relation_layer = annotate_relation_layer_from_files(
    doc=doc,
    assignments_path=RELATION_ASSIGNMENTS_PATH,
    cluster_assertions_path=CLUSTER_ASSERTIONS_PATH,
    force=True,
)

print(relation_layer.summary())
```

#### Sanity check

```python
coref = require_coref_layer(doc)
relations = require_relation_layer(doc)

for assignment_id in list(relations.assignments)[:20]:
    source = relations.source_cluster(coref, assignment_id)
    target = relations.target_cluster(coref, assignment_id)
    predicate = relations.predicate_span(doc, assignment_id)
    prop = relations.chosen_object_property_iri(assignment_id)
    conf = relations.confidence(assignment_id)

    print(
        f"{source.canonical_name} -- {predicate.text} / {prop} "
        f"({conf:.3f}) --> {target.canonical_name}"
    )
```

#### ABox population

```python
for assertion in relation_layer.cluster_assertions.values():
    assert_cluster_object_property(
        onto=onto,
        individual_ns=INDIVIDUAL_NS,
        source_cluster_id=assertion.source_cluster_id,
        object_property_iri=assertion.object_property_iri,
        target_cluster_id=assertion.target_cluster_id,
    )

save_ontology(
    onto,
    POPULATED_ONTOLOGY_PATH,
    document_format="turtle",
)
```

#### Save Doc

```python
save_doc(doc, RELATION_DOC_PATH)
```

---

## 4. Implementation boundaries and known V1 limitations

### Passive constructions

The extractor detects `is_passive`, but V1 does not invert source/target automatically. This preserves the behavior of the provided prototype and avoids hidden semantic rewriting.

### Complex OWL domain/range expressions

The router only uses named OWLClass domains/ranges in V1. If the ontology later uses complex class expressions as property domains/ranges, routing must be extended.

### One assertion per cluster pair

Aggregation V1 emits one best object property per `(source_cluster_id, target_cluster_id)` pair. A future version may allow multiple properties between the same pair if their support is strong.

### JSONL vs CSV

- Stage 1 is JSONL because candidate properties are nested.
- Stage 2 and stage 3 are CSV because they are flat and convenient to inspect.
- The final Doc layer stores Python objects, not verbose text snapshots.

---

## 5. Development checklist

1. Copy the generated `relationship_extraction/` package into the project.
2. Replace or merge `ontology/ontology_managment.py` with the generated extended version.
3. Add the imports to the notebook `# Imports` section.
4. Register `relation_layer` in `load_doc(...)`.
5. Add relation output paths in `## I/O config`.
6. Build `relation_catalog` and `relation_router` after `load_tbox(...)`.
7. Replace notebook-local edge extraction code with the stage calls.
8. Run stage 1 and inspect `routed_relation_candidates.jsonl`.
9. Run stage 2 and inspect `relation_assignments.csv`.
10. Run stage 3 and inspect `cluster_assertions.csv`.
11. Annotate `doc._.relation_layer`.
12. Run sanity checks.
13. Populate the ABox from `ClusterAssertion`.
