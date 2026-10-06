"""Offline contract checks for source selection and generated outputs."""

import unittest

from pydantic import TypeAdapter, ValidationError

from src.schemas.interview import InterviewResult, QAValidationResult
from src.schemas.presentation import PresentationPlan
from src.schemas.requests import DocumentInput, GenerationRequest, PresentationRequest


def source():
    return dict(kind="document", source_id="s1", document_id="d1", filename="notes.pdf",
                page=1, chunk_id="c1", excerpt="Python type hints describe expected types.")


def interview():
    return dict(topic="Type hints", source_mode="document", sources=[source()], items=[dict(
        question="What do type hints describe?", category="Python", difficulty="beginner",
        question_type="conceptual", answer="Expected types.", key_points=["Type contracts"],
        source_ids=["s1"], confidence="high",
    )])


def presentation():
    slides = [dict(number=1, layout="cover", purpose="Introduce topic", title="Type hints")]
    slides += [dict(number=i, layout="concept", purpose="Explain a concept", title=f"Concept {i}",
                    bullets=["Type hints describe types."], source_ids=["s1"]) for i in range(2, 11)]
    return dict(topic="Type hints", day=15, source_mode="document", template_id="reference",
                slides=slides, sources=[source()])


class SchemaTests(unittest.TestCase):
    def test_multiple_documents_and_request_dispatch(self):
        request = TypeAdapter(GenerationRequest).validate_python(dict(
            output="presentation", topic="Python", day=15, audience="Developers",
            template_id="reference", source=dict(mode="document", document_ids=["d1", "d2"]),
        ))
        self.assertIsInstance(request, PresentationRequest)
        self.assertEqual(request.source.document_ids, ["d1", "d2"])

    def test_invalid_document_selection(self):
        for ids in ([], [" "], ["d1", "d1"]):
            with self.assertRaises(ValidationError):
                DocumentInput(document_ids=ids)

    def test_web_mode_rejects_document_ids(self):
        with self.assertRaises(ValidationError):
            TypeAdapter(GenerationRequest).validate_python(dict(
                output="presentation", topic="Python", day=1, audience="Developers",
                template_id="reference", source=dict(mode="web", document_ids=["d1"]),
            ))

    def test_valid_results_roundtrip(self):
        for cls, payload in ((InterviewResult, interview()), (PresentationPlan, presentation())):
            value = cls.model_validate(payload)
            self.assertEqual(cls.model_validate_json(value.model_dump_json()), value)
            self.assertIn("properties", cls.model_json_schema())

    def test_unknown_citation_and_source_mode(self):
        for mutation in ("unknown", "mode", "duplicate_source", "duplicate_question"):
            value = interview()
            if mutation == "unknown": value["items"][0]["source_ids"] = ["invented"]
            elif mutation == "mode": value["source_mode"] = "web"
            elif mutation == "duplicate_source": value["sources"].append(source())
            else: value["items"].append(value["items"][0].copy())
            with self.assertRaises(ValidationError):
                InterviewResult.model_validate(value)

    def test_presentation_integrity(self):
        for mutation in ("count", "order", "cover", "evidence", "content", "code"):
            value = presentation()
            if mutation == "count": value["slides"].pop()
            elif mutation == "order": value["slides"][1]["number"] = 3
            elif mutation == "cover": value["slides"][1]["layout"] = "cover"
            elif mutation == "evidence": value["slides"][1]["source_ids"] = []
            elif mutation == "content": value["slides"][1]["bullets"] = []
            else: value["slides"][1]["layout"] = "code"
            with self.assertRaises(ValidationError):
                PresentationPlan.model_validate(value)

    def test_one_based_page_numbers(self):
        value = interview()
        value["sources"][0]["page"] = 0
        with self.assertRaises(ValidationError):
            InterviewResult.model_validate(value)

    def test_verdict_consistency(self):
        QAValidationResult(approved=True)
        QAValidationResult(approved=False, issues=["Unsupported claim"])
        for value in (dict(approved=True, issues=["Unresolved"]), dict(approved=False), dict(approved="yes")):
            with self.assertRaises(ValidationError):
                QAValidationResult.model_validate(value)


if __name__ == "__main__":
    unittest.main()
