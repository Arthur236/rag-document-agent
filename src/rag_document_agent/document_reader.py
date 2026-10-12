import re
from pathlib import Path

import pypdf
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx"}


class DocumentReader:
    """Read a document (PDF, Word, PowerPoint, or plain text) and expose its extracted text as paragraphs."""

    # initializing our document reader class
    def __init__(self, path=""):
        """
        starting our class Reader which has to take the path of the document we want to read.
        the file type is inferred from the extension.
        """
        self.path = path
        self.extension = Path(path).suffix.lower()
        if self.extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{self.extension}'. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        self.pages_text = ""

    # extracting from the document to text
    def extract_text(self):
        """
        extracting the content of the document into a string that we can later use as we wish
        """
        if self.extension == ".pdf":
            raw_text = self._extract_pdf_text()
        elif self.extension in (".txt", ".md"):
            raw_text = self._extract_plain_text()
        elif self.extension == ".docx":
            raw_text = self._extract_docx_text()
        elif self.extension == ".pptx":
            raw_text = self._extract_pptx_text()

        pdf_text = self._normalize_whitespace(raw_text)
        print(f"Successfully extracted text. Total characters: {len(pdf_text)}")
        self.pages_text = pdf_text
        return pdf_text

    def _extract_pdf_text(self) -> str:
        reader = pypdf.PdfReader(self.path)
        all_pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Only add if text extraction was successful
                all_pages_text.append(page_text)
        return "\n".join(all_pages_text)

    def _extract_plain_text(self) -> str:
        return Path(self.path).read_text(encoding="utf-8", errors="ignore")

    def _extract_docx_text(self) -> str:
        doc = DocxDocument(self.path)
        blocks = [p.text for p in doc.paragraphs if p.text.strip()]

        # tables aren't part of doc.paragraphs, so pull their text separately
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    blocks.append(row_text)

        return "\n\n".join(blocks)

    def _extract_pptx_text(self) -> str:
        presentation = Presentation(self.path)
        slides_text = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            slide_blocks = []
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    slide_blocks.append(shape.text_frame.text.strip())
                elif shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(
                            cell.text.strip() for cell in row.cells if cell.text.strip()
                        )
                        if row_text:
                            slide_blocks.append(row_text)
            if slide_blocks:
                slides_text.append(f"Slide {slide_number}:\n" + "\n".join(slide_blocks))
        return "\n\n".join(slides_text)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(
            r"\n([ \t]*\n)?[ \t]{2,}", "¶", text
        )  # 1. mark paragraph breaks BEFORE space normalization
        text = re.sub(r" +", " ", text)  # 2. collapse multiple spaces
        text = re.sub(
            r"[ \t]*\n[ \t]*", " ", text
        )  # 3. collapse word-wrap newlines → space
        text = text.replace("¶", "\n")  # 4. restore paragraph breaks as \n
        text = re.sub(r" +", " ", text).strip()  # 5. final space cleanup
        return text

    def extract_small_portion(self, min=0, max=None):
        """
        showing a portion of the document to allow the user to see what it looks like
        """
        if self.pages_text == "":
            self.extract_text()
        return self.pages_text[min:max]

    def get_chunks(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
        """
        splits the extracted document text into overlapping chunks using LangChain's
        RecursiveCharacterTextSplitter, which prefers to split on paragraph,
        then sentence, then word boundaries before falling back to a hard cut.
        call extract_text() first before using this method.
        """
        if self.pages_text == "":
            self.extract_text()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(self.pages_text)


###############
# example to Run to make sure the class is good
###############

# document_reader = DocumentReader(path="./data/Microsoft_10K_2025.pdf")
# document_reader.extract_text()
# print(document_reader.extract_small_portion(max=10000))
# print(document_reader.get_chunks()[:3])
