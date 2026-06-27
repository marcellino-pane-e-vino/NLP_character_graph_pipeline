from __future__ import annotations

try:
    from spacy.tokens import Doc
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "annotation_layer.spacy_extension requires spaCy. Install it with: pip install spacy"
    ) from exc

from annotation_layer.annotation_layer import AnnotationLayer


def register_spacy_annotation_extension(*, force: bool = False) -> None:
    if Doc.has_extension("annotation_layer"):
        if force:
            Doc.set_extension("annotation_layer", default=None, force=True)
        return
    Doc.set_extension("annotation_layer", default=None)


def ensure_annotation_layer(doc: Doc, *, document_id: str) -> AnnotationLayer:
    register_spacy_annotation_extension()
    if doc._.annotation_layer is None:
        doc._.annotation_layer = AnnotationLayer(document_id=document_id)
    return doc._.annotation_layer


def require_annotation_layer(doc: Doc) -> AnnotationLayer:
    if not Doc.has_extension("annotation_layer") or doc._.annotation_layer is None:
        raise ValueError("This Doc has no annotation_layer.")
    return doc._.annotation_layer


def require_entities(doc: Doc):
    return require_annotation_layer(doc).require_entities()


def require_relations(doc: Doc):
    return require_annotation_layer(doc).require_relations()
