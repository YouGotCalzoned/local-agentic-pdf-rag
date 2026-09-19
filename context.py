def build_context(results):
    return "\n\n".join(result.page_content for result in results)


def build_labeled_context(results):
    chunks = []
    for index, result in enumerate(results, start=1):
        source_id = f"S{index}"
        page = result.metadata.get("page")
        source = result.metadata.get("source")
        chunks.append(
            f"[{source_id}]\nSOURCE: {source}\nPAGE: {page}\n\n{result.page_content}"
        )
    return "\n\n".join(chunks)


def build_source_map(results):
    lines = []
    for index, result in enumerate(results, start=1):
        source_id = f"S{index}"
        page = result.metadata.get("page")
        source = result.metadata.get("source")
        lines.append(f"[{source_id}] Page {page} - {source}")
    return lines
