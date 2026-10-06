from src.ingestion.multimodal_parser import (
    MultiModalParser,
    ModalityType,
    ExtractedTable,
    BoundingBox
)


def test_parse_table_from_raw():
    parser = MultiModalParser()
    raw_matrix = [
        ["Quarter", "Revenue (M$)", "Operating Margin", "YoY Growth"],
        ["Q1 2024", "$120.5", "24.2%", "+18%"],
        ["Q2 2024", "$134.8", "25.1%", "+21%"],
        ["Q3 2024", "$148.2", "26.0%", "+25%"]
    ]

    table = parser.parse_table_from_raw(
        raw_rows=raw_matrix,
        caption="Fiscal Year 2024 Financial Performance",
        page=2,
        bbox=BoundingBox(page=2, x0=0.1, y0=0.2, x1=0.9, y1=0.6)
    )

    assert table.row_count == 3
    assert table.column_count == 4
    assert table.headers == ["Quarter", "Revenue (M$)", "Operating Margin", "YoY Growth"]
    assert "| Quarter | Revenue (M$) | Operating Margin | YoY Growth |" in table.markdown_content
    assert "| Q1 2024 | $120.5 | 24.2% | +18% |" in table.markdown_content
    assert table.bbox.page == 2


def test_parse_markdown_tables():
    parser = MultiModalParser()
    sample_text = """
    Below is the cloud pricing matrix:

    | Instance Type | vCPU | RAM (GB) | Hourly Rate ($) |
    | --- | --- | --- | --- |
    | c6g.large | 2 | 4 | 0.068 |
    | m6g.xlarge | 4 | 16 | 0.154 |

    Please review before provisioning.
    """

    tables = parser.parse_markdown_tables(sample_text, page=1)
    assert len(tables) == 1
    tbl = tables[0]
    assert tbl.row_count == 2
    assert tbl.column_count == 4
    assert tbl.headers == ["Instance Type", "vCPU", "RAM (GB)", "Hourly Rate ($)"]


def test_process_document_multimodal():
    parser = MultiModalParser()
    raw_tables = [
        [
            ["Metric", "Target", "Actual"],
            ["Uptime", "99.99%", "99.995%"],
            ["p95 Latency", "< 50ms", "42ms"]
        ]
    ]
    visual_figures = [
        {
            "figure_type": "architectural_diagram",
            "caption": "Multi-Tenant VPC Peering Topology",
            "summary": "Shows isolated tenant subnets communicating via AWS Transit Gateway with WAF inspection.",
            "ocr_detected_text": ["Transit Gateway", "Tenant VPC A", "Tenant VPC B", "Stateful Firewall"],
            "page": 1
        }
    ]

    content = """
    Executive Overview of Platform Infrastructure.
    All tenant data is strictly isolated across dedicated cryptographic domains.

    System telemetry confirms SLA compliance across all regions.
    """

    doc = parser.process_document(
        tenant_id="tenant_fintech_corp",
        title="2024 Infrastructure SLA & Topology Audit",
        content=content,
        raw_tables=raw_tables,
        visual_figures=visual_figures,
        clearance="CONFIDENTIAL"
    )

    assert doc.tenant_id == "tenant_fintech_corp"
    assert doc.tables_count == 1
    assert doc.figures_count == 1
    assert doc.text_blocks_count >= 1

    # Verify chunk modalities
    modalities = {c.modality for c in doc.chunks}
    assert ModalityType.TEXT in modalities
    assert ModalityType.TABLE_MARKDOWN in modalities
    assert ModalityType.VISUAL_FIGURE in modalities

    # Check table chunk has structured headers injected
    table_chunk = next(c for c in doc.chunks if c.modality == ModalityType.TABLE_MARKDOWN)
    assert "Schema: Columns (3): Metric, Target, Actual" in table_chunk.content
    assert table_chunk.structured_data["row_count"] == 2

    # Check visual figure chunk
    figure_chunk = next(c for c in doc.chunks if c.modality == ModalityType.VISUAL_FIGURE)
    assert "Transit Gateway" in figure_chunk.content
    assert "Vision AI Summary:" in figure_chunk.content


if __name__ == "__main__":
    test_parse_table_from_raw()
    test_parse_markdown_tables()
    test_process_document_multimodal()
    print("ALL TESTS PASSED SUCCESSFULLY!")
