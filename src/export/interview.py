"""Readable interview output; consumes only a validated result."""

from src.schemas.interview import InterviewResult


def render_interview_markdown(result: InterviewResult) -> str:
    lines = [f"# {result.topic}"]
    for number, item in enumerate(result.items, 1):
        lines.extend(["", f"## {number}. {item.question}", "", item.answer, "", "Key points:"])
        lines.extend(f"- {point}" for point in item.key_points)
        if item.follow_up_questions:
            lines.extend(["", "Follow-up questions:"])
            lines.extend(f"- {question}" for question in item.follow_up_questions)
        lines.extend(["", f"Confidence: {item.confidence}", "Sources: " + ", ".join(item.source_ids)])
    lines.extend(["", "## Sources"])
    for source in result.sources:
        if source.kind == "document":
            location = f"{source.filename}, page {source.page}"
        else:
            location = str(source.url)
        lines.append(f"- {source.source_id}: {location}")
    return "\n".join(lines) + "\n"
