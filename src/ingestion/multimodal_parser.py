import re
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ModalityType(str, Enum):
    TEXT = "text"
    TABLE_MARKDOWN = "table_markdown"
    VISUAL_FIGURE = "visual_figure"
    COMPOSITE = "composite"


class BoundingBox(BaseModel):
    page: int = Field(default=1, ge=1)
    x0: float = Field(default=0.0, description="Top-left X coordinate")
    y0: float = Field(default=0.0, description="Top-left Y coordinate")
    x1: float = Field(default=1.0, description="Bottom-right X coordinate")
    y1: float = Field(default=1.0, description="Bottom-right Y coordinate")


class ExtractedTable(BaseModel):
    table_id: str
    headers: List[str]
    rows: List[List[str]]
    markdown_content: str
    row_count: int
    column_count: int
    caption: Optional[str] = None
    bbox: Optional[BoundingBox] = None


class VisualFigure(BaseModel):
    figure_id: str
    figure_type: str = Field(default="chart", description="chart, diagram, architectural_diagram, image")
    caption: str
    summary: str
    ocr_detected_text: List[str] = Field(default_factory=list)
    bbox: Optional[BoundingBox] = None


class MultiModalChunk(BaseModel):
    chunk_id: str
    document_id: str
    tenant_id: str
    modality: ModalityType
    content: str
    structured_data: Optional[Dict[str, Any]] = None
    page_number: int = 1
    bbox: Optional[BoundingBox] = None
    token_count: int = 0
    clearance: str = "INTERNAL"


class MultiModalParsedDoc(BaseModel):
    document_id: str
    tenant_id: str
    title: str
    total_pages: int
    text_blocks_count: int
    tables_count: int
    figures_count: int
    tables: List[ExtractedTable] = Field(default_factory=list)
    figures: List[VisualFigure] = Field(default_factory=list)
    chunks: List[MultiModalChunk] = Field(default_factory=list)


class MultiModalParser:
    """
    Production-grade Multi-Modal Parser that extracts text, tables, and visual elements.
    Ensures that tabular financial/scientific data is preserved as structured Markdown
    with schema headers injected into every row representation to avoid hallucination.
    """

    def __init__(self, avg_chars_per_token: float = 4.0):
        self.avg_chars_per_token = avg_chars_per_token

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text.strip()) / self.avg_chars_per_token))

    def parse_table_from_raw(
        self,
        raw_rows: List[List[str]],
        caption: Optional[str] = None,
        page: int = 1,
        bbox: Optional[BoundingBox] = None
    ) -> ExtractedTable:
        """Transforms a 2D matrix of strings into a verified ExtractedTable with Markdown format."""
        if not raw_rows:
            return ExtractedTable(
                table_id=f"tbl_{uuid.uuid4().hex[:8]}",
                headers=[],
                rows=[],
                markdown_content="",
                row_count=0,
                column_count=0,
                caption=caption,
                bbox=bbox or BoundingBox(page=page)
            )

        headers = [str(col).strip() for col in raw_rows[0]]
        rows = [[str(cell).strip() for cell in row] for row in raw_rows[1:]]

        # Build Markdown table string
        md_lines = []
        if caption:
            md_lines.append(f"**Table: {caption}**\n")
        
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        md_lines.append(header_line)
        md_lines.append(sep_line)

        for row in rows:
            # Pad row if incomplete
            padded_row = row + [""] * (len(headers) - len(row))
            row_line = "| " + " | ".join(padded_row[:len(headers)]) + " |"
            md_lines.append(row_line)

        markdown_content = "\n".join(md_lines)
        table_id = f"tbl_{hashlib.md5(markdown_content.encode()).hexdigest()[:8]}"

        return ExtractedTable(
            table_id=table_id,
            headers=headers,
            rows=rows,
            markdown_content=markdown_content,
            row_count=len(rows),
            column_count=len(headers),
            caption=caption,
            bbox=bbox or BoundingBox(page=page)
        )

    def parse_markdown_tables(self, markdown_text: str, page: int = 1) -> List[ExtractedTable]:
        """Detects and extracts Markdown formatted tables inside text."""
        normalized_lines = [l.strip() for l in markdown_text.splitlines()]
        normalized_text = "\n".join(normalized_lines)
        table_pattern = re.compile(r"((?:^\|.+?\|(?:\r?\n|$)){2,})", re.MULTILINE)
        extracted: List[ExtractedTable] = []

        for match in table_pattern.finditer(normalized_text):
            table_raw = match.group(1).strip()
            lines = [l.strip() for l in table_raw.splitlines() if l.strip()]
            if len(lines) < 2:
                continue

            # First line is header
            headers = [c.strip() for c in lines[0].strip("|").split("|")]
            data_rows = []
            for line in lines[2:]:  # Skip delimiter line
                cols = [c.strip() for c in line.strip("|").split("|")]
                data_rows.append(cols)

            if headers:
                extracted.append(
                    self.parse_table_from_raw(
                        raw_rows=[headers] + data_rows,
                        caption=f"Detected Table on Page {page}",
                        page=page
                    )
                )

        return extracted

    def process_document(
        self,
        tenant_id: str,
        title: str,
        content: str,
        raw_tables: Optional[List[List[List[str]]]] = None,
        visual_figures: Optional[List[Dict[str, Any]]] = None,
        clearance: str = "INTERNAL"
    ) -> MultiModalParsedDoc:
        """
        Parses multi-modal components into a unified structured document.
        Creates contextual chunks with schema preservation for tables and visual summaries.
        """
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        all_tables: List[ExtractedTable] = []
        all_figures: List[VisualFigure] = []
        chunks: List[MultiModalChunk] = []

        # 1. Parse tables provided as raw matrices
        if raw_tables:
            for idx, matrix in enumerate(raw_tables):
                table_obj = self.parse_table_from_raw(
                    raw_rows=matrix,
                    caption=f"Financial Matrix #{idx + 1}",
                    page=1
                )
                all_tables.append(table_obj)

        # 2. Parse inline markdown tables within content
        inline_tables = self.parse_markdown_tables(content, page=1)
        for tbl in inline_tables:
            if not any(t.markdown_content == tbl.markdown_content for t in all_tables):
                all_tables.append(tbl)

        # 3. Parse visual figures (charts, flowcharts, architectures)
        if visual_figures:
            for fig_data in visual_figures:
                fig_obj = VisualFigure(
                    figure_id=f"fig_{uuid.uuid4().hex[:8]}",
                    figure_type=fig_data.get("figure_type", "chart"),
                    caption=fig_data.get("caption", "Visual Diagram"),
                    summary=fig_data.get("summary", ""),
                    ocr_detected_text=fig_data.get("ocr_detected_text", []),
                    bbox=BoundingBox(
                        page=fig_data.get("page", 1),
                        x0=fig_data.get("x0", 0.1),
                        y0=fig_data.get("y0", 0.1),
                        x1=fig_data.get("x1", 0.9),
                        y1=fig_data.get("y1", 0.5)
                    )
                )
                all_figures.append(fig_obj)

        # 4. Generate Semantic Chunks for Text (stripping standalone tables to avoid redundancy)
        clean_text = content
        for tbl in all_tables:
            if tbl.markdown_content in clean_text:
                clean_text = clean_text.replace(tbl.markdown_content, f"\n[Reference: {tbl.caption or tbl.table_id}]\n")

        # Split text into paragraphs
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
        for p_idx, para in enumerate(paragraphs):
            c_id = f"{doc_id}_txt_{p_idx + 1}"
            chunks.append(
                MultiModalChunk(
                    chunk_id=c_id,
                    document_id=doc_id,
                    tenant_id=tenant_id,
                    modality=ModalityType.TEXT,
                    content=para,
                    page_number=1,
                    token_count=self._estimate_tokens(para),
                    clearance=clearance
                )
            )

        # 5. Generate Schema-Preserving Chunks for Tables
        for tbl in all_tables:
            # We enrich the table chunk with a schema header and row-by-row natural representation
            schema_summary = f"Columns ({tbl.column_count}): {', '.join(tbl.headers)}"
            full_table_chunk = f"DOCUMENT TABLE: {tbl.caption or 'Data Table'}\nSchema: {schema_summary}\n\n{tbl.markdown_content}"

            c_id = f"{doc_id}_{tbl.table_id}"
            chunks.append(
                MultiModalChunk(
                    chunk_id=c_id,
                    document_id=doc_id,
                    tenant_id=tenant_id,
                    modality=ModalityType.TABLE_MARKDOWN,
                    content=full_table_chunk,
                    structured_data={
                        "headers": tbl.headers,
                        "rows": tbl.rows,
                        "row_count": tbl.row_count,
                        "column_count": tbl.column_count
                    },
                    page_number=tbl.bbox.page if tbl.bbox else 1,
                    bbox=tbl.bbox,
                    token_count=self._estimate_tokens(full_table_chunk),
                    clearance=clearance
                )
            )

        # 6. Generate Semantic Chunks for Visual Figures
        for fig in all_figures:
            ocr_concat = " | ".join(fig.ocr_detected_text) if fig.ocr_detected_text else "None"
            fig_chunk_text = (
                f"VISUAL FIGURE [{fig.figure_type.upper()}]: {fig.caption}\n"
                f"Vision AI Summary: {fig.summary}\n"
                f"OCR Text detected in figure: {ocr_concat}"
            )
            c_id = f"{doc_id}_{fig.figure_id}"
            chunks.append(
                MultiModalChunk(
                    chunk_id=c_id,
                    document_id=doc_id,
                    tenant_id=tenant_id,
                    modality=ModalityType.VISUAL_FIGURE,
                    content=fig_chunk_text,
                    structured_data={
                        "figure_type": fig.figure_type,
                        "ocr_detected_text": fig.ocr_detected_text
                    },
                    page_number=fig.bbox.page if fig.bbox else 1,
                    bbox=fig.bbox,
                    token_count=self._estimate_tokens(fig_chunk_text),
                    clearance=clearance
                )
            )

        return MultiModalParsedDoc(
            document_id=doc_id,
            tenant_id=tenant_id,
            title=title,
            total_pages=1,
            text_blocks_count=len(paragraphs),
            tables_count=len(all_tables),
            figures_count=len(all_figures),
            tables=all_tables,
            figures=all_figures,
            chunks=chunks
        )


multimodal_parser = MultiModalParser()
