"""Public ontology API for the NLP character graph pipeline.

Import from this module in notebooks and pipeline orchestration code. Concrete
implementation is split across ``io``, ``tbox``, ``relations``, and ``abox``.
"""

from __future__ import annotations

from ontology.io import (
    load_ontology,
    save_ontology,
)
from ontology.tbox import (
    build_class_graph,
    load_tbox,
)
from ontology.relations import (
    RelationPropertySpec,
    RelationRouter,
    build_relation_router,
)
from ontology.abox import (
    cluster_individual,
    assert_cluster_type,
    assert_cluster_relation,
    assert_cluster_value,
)


__all__ = [
    "load_ontology",
    "save_ontology",
    "build_class_graph",
    "load_tbox",
    "RelationPropertySpec",
    "RelationRouter",
    "build_relation_router",
    "cluster_individual",
    "assert_cluster_type",
    "assert_cluster_relation",
    "assert_cluster_value",
]
