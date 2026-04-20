"""
MCP server for extracting structured alloy datasets from MinerU-parsed patents.

Design principle: parsing happens on the server side. The LLM receives
ready-to-write dataset rows, not raw OCR text, so the context window stays small
and the extraction is deterministic.

Directory layout expected:
    patents/
        patent1/
            patent_content_list.json   # MinerU output (required)
            patent_middle.json         # optional, not read here
            patent.md                  # optional, not read here
        patent2/
            ...
"""

from __future__ import annotations

import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP

mcp = FastMCP("PatentServer")

PATENTS_DIR = Path("patents")
OUTPUT_DIR = Path("datasets")  # CSV files are written here


# ---------------------------------------------------------------------------
# Russian to element symbol mapping
# ---------------------------------------------------------------------------

RUSSIAN_ELEMENTS = {
    'углерод': 'C',
    'хром': 'Cr',
    'кобальт': 'Co',
    'вольфрам': 'W',
    'молибден': 'Mo',
    'титан': 'Ti',
    'алюминий': 'Al',
    'ниобий': 'Nb',
    'тантал': 'Ta',
    'гафний': 'Hf',
    'рений': 'Re',
    'бор': 'B',
    'цирконий': 'Zr',
    'магний': 'Mg',
    'железо': 'Fe',
    'марганец': 'Mn',
    'кремний': 'Si',
    'никель': 'Ni',
    'церий': 'Ce',
    'медь': 'Cu',
    'ванадий': 'V',
    'азот': 'N',
    'кислород': 'O',
    'фосфор': 'P',
    'сера': 'S',
}

ENGLISH_ELEMENTS = {
    'carbon': 'C',
    'chromium': 'Cr',
    'cobalt': 'Co',
    'tungsten': 'W',
    'molybdenum': 'Mo',
    'titanium': 'Ti',
    'aluminum': 'Al',
    'aluminium': 'Al',
    'niobium': 'Nb',
    'tantalum': 'Ta',
    'hafnium': 'Hf',
    'rhenium': 'Re',
    'boron': 'B',
    'zirconium': 'Zr',
    'magnesium': 'Mg',
    'iron': 'Fe',
    'manganese': 'Mn',
    'silicon': 'Si',
    'nickel': 'Ni',
    'cerium': 'Ce',
    'copper': 'Cu',
    'vanadium': 'V',
    'nitrogen': 'N',
    'oxygen': 'O',
    'phosphorus': 'P',
    'sulfur': 'S',
}

ELEMENT_MAP = {**RUSSIAN_ELEMENTS, **ENGLISH_ELEMENTS}


def _russian_to_element(russian_name: str) -> str:
    """Convert Russian or English element name to symbol."""
    name = russian_name.lower().strip()
    # Remove common prefixes like numbers and dashes
    name = re.sub(r'^\d+\s*', '', name)
    name = re.sub(r'^[-–]\s*', '', name)
    return ELEMENT_MAP.get(name, name)


# ---------------------------------------------------------------------------
# HTML table parsing
# ---------------------------------------------------------------------------

class _TableParser(HTMLParser):
    """Parses a single <table>...</table> fragment into a list of rows (list[list[str]])."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: List[str] = []
        self._cell: List[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._row.append("".join(self._cell).strip())
            self._in_cell = False
        elif tag == "tr":
            if self._row:
                self.rows.append(self._row)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell.append(data)


def _parse_html_table(html: str) -> List[List[str]]:
    p = _TableParser()
    p.feed(html)
    return p.rows


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------

def _clean_cell(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _parse_numeric(s: str) -> Optional[float]:
    """Extract a single float from a cell like '18.17', '1.2-1.6%', 'max. 0.08', '≥1034MPa'."""
    s = _clean_cell(s)
    if not s:
        return None
    # ranges like 17-21% -> take midpoint
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def _parse_range(s: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Return (low, high) for range strings. 'max. 0.08' -> (None, 0.08). '17-21%' -> (17, 21)."""
    s = _clean_cell(s)
    if not s:
        return None
    if re.search(r"max\.?", s, re.I):
        v = _parse_numeric(s)
        return (None, v) if v is not None else None
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    v = _parse_numeric(s)
    return (v, v) if v is not None else None


# ---------------------------------------------------------------------------
# MinerU content_list loading and table selection
# ---------------------------------------------------------------------------

def _load_content_list(patent_dir: Path) -> List[Dict[str, Any]]:
    path = patent_dir / "patent_content_list.json"
    if not path.exists():
        cands = list(patent_dir.glob("*_content_list.json"))
        if not cands:
            raise FileNotFoundError(f"No content_list.json in {patent_dir}")
        path = cands[0]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_markdown(patent_dir: Path) -> Optional[str]:
    """Load patent.md file if it exists."""
    md_path = patent_dir / "patent.md"
    if not md_path.exists():
        cands = list(patent_dir.glob("*.md"))
        if not cands:
            return None
        md_path = cands[0]
    try:
        with md_path.open("r", encoding="utf-8") as f:
            return f.read()
    except:
        return None


def _find_table_by_caption(items: List[Dict[str, Any]], pattern: str) -> Optional[List[List[str]]]:
    rx = re.compile(pattern, re.I)
    for it in items:
        if it.get("type") != "table":
            continue
        caps = it.get("table_caption") or []
        if any(rx.search(c or "") for c in caps):
            return _parse_html_table(it.get("table_body", ""))
    return None


def _find_composition_tables(items: List[Dict[str, Any]]) -> List[List[List[str]]]:
    """Tables whose first row header looks like Alloy | Ni | Fe | Cr | Mo | Ti | Al | Nb+Ta | Co."""
    results = []
    for it in items:
        if it.get("type") != "table":
            continue
        rows = _parse_html_table(it.get("table_body", ""))
        if not rows:
            continue
        header = next((r for r in rows if any(c.strip() for c in r)), None)
        if not header:
            continue
        header_clean = [_clean_cell(c).lower() for c in header]
        if "alloy" in header_clean and "ni" in header_clean and "cr" in header_clean:
            results.append(rows)
    return results


def _find_atomic_table(items: List[Dict[str, Any]]) -> Optional[List[List[str]]]:
    for it in items:
        if it.get("type") != "table":
            continue
        caps = it.get("table_caption") or []
        if any("6a" in (c or "") for c in caps):
            return _parse_html_table(it.get("table_body", ""))
        rows = _parse_html_table(it.get("table_body", ""))
        for r in rows:
            hdr = [_clean_cell(c).lower().replace(" ", "") for c in r]
            if "al/ti" in hdr and "al+ti" in hdr:
                return rows
    return None


def _find_solvus_table(items: List[Dict[str, Any]]) -> Optional[List[List[str]]]:
    """Table 6b — has columns delta-solv, gamma-solv, dT, 10HV, n-phase."""
    for it in items:
        if it.get("type") != "table":
            continue
        body = it.get("table_body", "")
        low = body.lower()
        if ("δ-solv" in body or "d-solv" in low or "δ-solv." in body) \
                and ("γ" in body or "y'-solv" in low or "gamma" in low) \
                and "10hv" in low:
            return _parse_html_table(body)
    return None


def _find_mechanical_a780_table(items: List[Dict[str, Any]]) -> Optional[List[List[str]]]:
    for it in items:
        if it.get("type") != "table":
            continue
        caps = it.get("table_caption") or []
        if any(re.search(r"table\s*8", c or "", re.I) for c in caps):
            return _parse_html_table(it.get("table_body", ""))
    return None


# ---------------------------------------------------------------------------
# Metadata extraction (supports both US and RU patents)
# ---------------------------------------------------------------------------

def _extract_metadata_us(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract metadata from US-style patent cover page."""
    md: Dict[str, Any] = {
        "patent_number": None,
        "title": None,
        "authors": [],
        "applicant": None,
        "assignee": None,
        "country": None,
        "filing_date": None,
        "publication_date": None,
        "priority_date": None,
    }
    texts = [it.get("text", "") for it in items if it.get("type") == "text" and it.get("page_idx") == 0]
    joined = "\n".join(texts)

    m = re.search(r"Pub\.\s*No\.\s*:\s*([A-Z]{2}\s*\d{4}/\d+\s*[A-Z0-9]*)", joined)
    if m:
        md["patent_number"] = re.sub(r"\s+", "", m.group(1))

    m = re.search(r"Pub\.\s*Date\s*:?\s*([A-Za-z]+\.?\s*\d{1,2},\s*\d{4})", joined)
    if m:
        md["publication_date"] = m.group(1).strip()

    m = re.search(r"Filed\s*:?\s*([A-Za-z]+\.?\s*\d{1,2},\s*\d{4})", joined)
    if m:
        md["filing_date"] = m.group(1).strip()

    m = re.search(r"Foreign Application Priority Data[^\n]*\n([A-Za-z]+\.?\s*\d{1,2},\s*\d{4})", joined)
    if m:
        md["priority_date"] = m.group(1).strip()

    m = re.search(r"\(54\)\s*([A-Z][A-Z \-]+?)(?:\n|\s{2,})", joined)
    if m:
        md["title"] = m.group(1).strip()
    else:
        if len(texts) >= 2 and re.match(r"^[A-Z][A-Z \-]+$", _clean_cell(texts[1])):
            md["title"] = _clean_cell(texts[1])

    m = re.search(r"\(71\)\s*Applicant\s*:\s*([^\n]+)", joined)
    if m:
        md["applicant"] = _clean_cell(m.group(1))

    m = re.search(r"\(73\)\s*Assignee\s*:\s*([^\n]+)", joined)
    if m:
        md["assignee"] = _clean_cell(m.group(1))

    m = re.search(r"\(72\)\s*Inventors?\s*:\s*(.+?)(?=\(73\)|$)", joined, re.S)
    if m:
        inv_raw = re.sub(r"\s+", " ", m.group(1))
        authors = []
        for piece in inv_raw.split(";"):
            piece = piece.strip().strip(",")
            if piece:
                authors.append(piece)
        md["authors"] = authors

    m = re.search(r"\(([A-Z]{2})\)", md.get("applicant") or "")
    if m:
        md["country"] = m.group(1)
    elif md["patent_number"] and md["patent_number"].startswith("US"):
        md["country"] = "US"

    return md


def _extract_metadata_ru(md_text: str) -> Dict[str, Any]:
    """Extract metadata from Russian patent markdown."""
    md: Dict[str, Any] = {
        "patent_number": None,
        "title": None,
        "authors": [],
        "applicant": None,
        "assignee": None,
        "country": "RU",
        "filing_date": None,
        "publication_date": None,
        "priority_date": None,
    }
    
    # Patent number
    m = re.search(r'RU\s*(\d{7})', md_text)
    if m:
        md["patent_number"] = f"RU{m.group(1)}C1"
    
    # Title (usually in all caps after the number)
    m = re.search(r'\(54\)\s*(.+?)(?:\n|$)', md_text)
    if m:
        md["title"] = m.group(1).strip()
    
    # Authors (after "Автор(ы):")
    m = re.search(r'\(72\)\s*Автор\(ы\):\s*(.+?)(?=\(73\)|$)', md_text, re.DOTALL)
    if m:
        authors_text = m.group(1).strip()
        authors = [a.strip() for a in re.split(r'[;,]', authors_text) if a.strip()]
        md["authors"] = authors
    
    # Applicant (after "Патентообладатель(и):")
    m = re.search(r'\(73\)\s*Патентообладатель\(и\):\s*(.+?)(?=\n\n|\Z)', md_text, re.DOTALL)
    if m:
        md["applicant"] = m.group(1).strip()
    
    # Dates
    m = re.search(r'Заявка:\s*\d+,\s*(\d{2}\.\d{2}\.\d{4})', md_text)
    if m:
        md["filing_date"] = m.group(1)
    
    m = re.search(r'Опубликовано:\s*(\d{2}\.\d{2}\.\d{4})', md_text)
    if m:
        md["publication_date"] = m.group(1)
    
    return md


def _extract_metadata(items: List[Dict[str, Any]], md_text: Optional[str] = None) -> Dict[str, Any]:
    """Extract metadata with fallback to markdown for RU patents."""
    md = _extract_metadata_us(items)
    
    if not md.get("patent_number") and md_text:
        ru_md = _extract_metadata_ru(md_text)
        if ru_md.get("patent_number"):
            md.update(ru_md)
    
    return md


# ---------------------------------------------------------------------------
# Improved Markdown composition parser for Russian patents
# ---------------------------------------------------------------------------

def _parse_md_composition_blocks_improved(md_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse composition blocks from Russian patent markdown.
    Returns dict with block names and their compositions.
    """
    blocks = {}
    lines = md_text.split('\n')
    
    current_block = {}
    current_name = None
    in_block = False
    
    # Pattern for element lines - matches: "Углерод - 0,06-0,13" or "Хром - 8,0-12,0"
    # Also matches lines with numbers at start: "10 Вольфрам - 5,2-5,9"
    element_pattern = re.compile(
        r'^(?:\d+\s+)?'  # optional leading number
        r'(Углерод|Хром|Кобальт|Вольфрам|Молибден|Титан|Алюминий|Ниобий|Тантал|'
        r'Гафний|Рений|Бор|Цирконий|Магний|Железо|Марганец|Кремний|Никель|Церий)'
        r'\s*[-–]\s*(\d+(?:[.,]\d+)?)(?:\s*[-–]\s*(\d+(?:[.,]\d+)?))?',
        re.IGNORECASE
    )
    
    # Pattern for block headers (e.g., "Пример", "Предлагается сплав")
    block_headers = ['пример', 'предлагается', 'прототип', 'известен', 'состава']
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this is a block header
        line_lower = line.lower()
        if any(header in line_lower for header in block_headers):
            # Save previous block
            if current_block and current_name:
                blocks[current_name] = current_block
            # Start new block
            current_block = {}
            current_name = line[:50]  # Use first 50 chars as name
            in_block = True
            i += 1
            continue
        
        # Parse element lines
        match = element_pattern.match(line)
        if match:
            in_block = True
            element_ru = match.group(1)
            element = _russian_to_element(element_ru)
            low_str = match.group(2).replace(',', '.')
            high_str = match.group(3).replace(',', '.') if match.group(3) else None
            
            if high_str:
                # Range - store as dict with min/max
                current_block[element] = {
                    'value': (float(low_str) + float(high_str)) / 2,  # midpoint
                    'min': float(low_str),
                    'max': float(high_str),
                    'type': 'range'
                }
            else:
                # Exact value
                current_block[element] = {
                    'value': float(low_str),
                    'type': 'exact'
                }
        
        # If we see "Никель - остальное", close the block
        if 'никель' in line_lower and 'остальное' in line_lower:
            if current_block and current_name:
                blocks[current_name] = current_block
                current_block = {}
                current_name = None
                in_block = False
        
        i += 1
    
    # Save last block
    if current_block and current_name:
        blocks[current_name] = current_block
    
    return blocks


def _parse_mechanical_properties_ru(md_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse mechanical properties table from Russian patent."""
    mech_data = {}
    
    # Ищем таблицу с механическими свойствами
    # Таблица имеет вид:
    # | Сплав | σB | σ0,2 | δ | ψ | σ0,2/100 | СРТУ |
    # | Заявляемый | 1635 | 1189 | 12,6 | 16,3 | 998 | 1,5·10⁻⁴ |
    # | Прототип | 1420 | 1088 | 7,5 | 7,8 | 830 | 8,2·10⁻⁴ |
    
    # Находим строку с заголовками
    header_pattern = r'σ[БB]\s*.*?σ0[,，]2\s*.*?δ\s*.*?ψ\s*.*?σ0[,，]2/\d+\s*.*?СРТУ'
    
    # Находим строки с числами после заголовка
    value_pattern = re.compile(
        r'(?:Заявляемый|Предлагаемый|Прототип)\s*\|?\s*(\d+(?:[.,]\d+)?)\s*\|?\s*(\d+(?:[.,]\d+)?)\s*\|?\s*(\d+(?:[.,]\d+)?)\s*\|?\s*(\d+(?:[.,]\d+)?)\s*\|?\s*(\d+(?:[.,]\d+)?)\s*\|?\s*([\d.,]+(?:[·×][\d⁻]+)?)',
        re.IGNORECASE
    )
    
    # Альтернативный паттерн для простых таблиц
    simple_pattern = re.compile(
        r'(\d+)[,\s]+(\d+)[,\s]+(\d+(?:[.,]\d+)?)[,\s]+(\d+(?:[.,]\d+)?)[,\s]+(\d+)[,\s]+([\d.,]+(?:[·×][\d⁻]+)?)',
        re.IGNORECASE
    )
    
    # Ищем секцию с таблицей
    # В вашем .md файле таблица находится после "представлены в таблице"
    table_section = re.search(r'представлены в таблице.*?(?=\n\n|\Z)', md_text, re.DOTALL | re.IGNORECASE)
    if not table_section:
        return mech_data
    
    table_text = table_section.group(0)
    
    # Ищем значения для заявляемого сплава
    proposed_match = re.search(r'Заявляемый[^\d]*(\d+(?:[.,]\d+)?)[^\d]*(\d+(?:[.,]\d+)?)[^\d]*(\d+(?:[.,]\d+)?)[^\d]*(\d+(?:[.,]\d+)?)[^\d]*(\d+(?:[.,]\d+)?)[^\d]*([\d.,]+(?:[·×][\d⁻]+)?)', table_text, re.IGNORECASE)
    if proposed_match:
        mech_data['proposed_alloy'] = {
            'UTS_MPa': float(proposed_match.group(1).replace(',', '.')),
            'YS_MPa': float(proposed_match.group(2).replace(',', '.')),
            'elongation_pct': float(proposed_match.group(3).replace(',', '.')),
            'reduction_area_pct': float(proposed_match.group(4).replace(',', '.')),
            'stress_rupture_100h_MPa': float(proposed_match.group(5).replace(',', '.')),
            'test_temperature_C': 650,
        }
        # Парсим СРТУ (может быть в формате 1,5·10⁻⁴)
        creep_str = proposed_match.group(6).replace(',', '.')
        if '·10' in creep_str:
            parts = creep_str.split('·10')
            base = float(parts[0])
            exp_str = parts[1].replace('⁻', '-').replace('⁴', '4').replace('⁵', '5')
            mech_data['proposed_alloy']['creep_rate'] = base * (10 ** int(exp_str))
        else:
            mech_data['proposed_alloy']['creep_rate'] = float(creep_str)
    
    # Ищем значения для прототипа
    proto_match = re.search(r'Прототип[^\d]*(\d+(?:[.,]\d+)?)[^\d]*(\d+(?:[.,]\d+)?)[^\d]*(\d+(?:[.,]\d+)?)[^\d]*(\d+(?:[.,]\d+)?)[^\d]*(\d+(?:[.,]\d+)?)[^\d]*([\d.,]+(?:[·×][\d⁻]+)?)', table_text, re.IGNORECASE)
    if proto_match:
        mech_data['prototype'] = {
            'UTS_MPa': float(proto_match.group(1).replace(',', '.')),
            'YS_MPa': float(proto_match.group(2).replace(',', '.')),
            'elongation_pct': float(proto_match.group(3).replace(',', '.')),
            'reduction_area_pct': float(proto_match.group(4).replace(',', '.')),
            'stress_rupture_100h_MPa': float(proto_match.group(5).replace(',', '.')),
            'test_temperature_C': 650,
        }
        creep_str = proto_match.group(6).replace(',', '.')
        if '·10' in creep_str:
            parts = creep_str.split('·10')
            base = float(parts[0])
            exp_str = parts[1].replace('⁻', '-').replace('⁴', '4')
            mech_data['prototype']['creep_rate'] = base * (10 ** int(exp_str))
        else:
            mech_data['prototype']['creep_rate'] = float(creep_str)
    
    return mech_data


def _build_composition_wt(items: List[Dict[str, Any]], md_text: Optional[str] = None) -> Dict[str, Dict[str, Optional[float]]]:
    """Merge all composition tables (wt%) across the patent. Falls back to markdown if no tables found."""
    merged: Dict[str, Dict[str, Optional[float]]] = {}
    
    # Try HTML tables first
    for rows in _find_composition_tables(items):
        header_idx = None
        for i, r in enumerate(rows):
            if any(_clean_cell(c).lower() == "alloy" for c in r):
                header_idx = i
                break
        if header_idx is None:
            continue
        header = [_clean_cell(c) for c in rows[header_idx]]
        data = rows[header_idx + 1:]
        elem_cols = [c for c in header[1:]]
        for r in data:
            if not r:
                continue
            name = _clean_cell(r[0])
            if not name or name.lower() in ("alloy", "element"):
                continue
            entry = merged.setdefault(name, {})
            for i, val in enumerate(r[1:], start=1):
                if i - 1 >= len(elem_cols):
                    continue
                elem = elem_cols[i - 1]
                v = _clean_cell(val)
                if v.lower() == "rest":
                    entry[elem] = "rest"
                else:
                    num = _parse_numeric(v)
                    entry[elem] = num
    
    # If no tables found or tables have garbage (translit), try markdown fallback
    if (not merged or len(merged) == 0) and md_text:
        md_blocks = _parse_md_composition_blocks_improved(md_text)
        for block_name, block_data in md_blocks.items():
            alloy_name = f"RU_{block_name[:30].replace(' ', '_')}"
            entry = {}
            for element, data in block_data.items():
                if isinstance(data, dict) and 'value' in data:
                    entry[element] = data['value']
                elif isinstance(data, (int, float)):
                    entry[element] = data
            if entry:
                merged[alloy_name] = entry
        
        # Also try to extract from explicit example (Углерод - 0,09 etc.)
        example_pattern = re.compile(
            r'Углерод\s*[-–]\s*(\d+(?:[.,]\d+)?).*?'
            r'Хром\s*[-–]\s*(\d+(?:[.,]\d+)?).*?'
            r'Кобальт\s*[-–]\s*(\d+(?:[.,]\d+)?)',
            re.DOTALL | re.IGNORECASE
        )
        example_match = example_pattern.search(md_text)
        if example_match:
            example_composition = {}
            # Parse all elements from example section
            example_section = re.search(r'Пример.*?Углерод.*?Никель.*?остальное', md_text, re.DOTALL | re.IGNORECASE)
            if example_section:
                for element_ru, element_sym in RUSSIAN_ELEMENTS.items():
                    pattern = re.compile(rf'{element_ru}\s*[-–]\s*(\d+(?:[.,]\d+)?)', re.IGNORECASE)
                    match = pattern.search(example_section.group(0))
                    if match:
                        example_composition[element_sym] = float(match.group(1).replace(',', '.'))
                if example_composition:
                    merged['RU_example_alloy'] = example_composition
    
    return merged


def _build_composition_at(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    rows = _find_atomic_table(items)
    if not rows:
        return {}
    header_idx = None
    for i, r in enumerate(rows):
        if any("al/ti" in _clean_cell(c).lower().replace(" ", "") for c in r):
            header_idx = i
            break
    if header_idx is None:
        return {}
    header = [_clean_cell(c) for c in rows[header_idx]]
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for r in rows[header_idx + 1:]:
        if not r:
            continue
        name = _clean_cell(r[0])
        if not name or name.lower() in ("alloy", "alloy at %"):
            continue
        entry: Dict[str, Optional[float]] = {}
        for i, val in enumerate(r[1:], start=1):
            if i < len(header):
                entry[header[i]] = _parse_numeric(val)
        if "Al/Ti" in header and entry.get("Al/Ti") is None and entry.get("Al") is None:
            break
        out[name] = entry
    return out


_ALLOY_NAME_RE = re.compile(r"^(?:[VL]\d{2}|A718(?:Plus)?|Waspaloy|RU_.*)$")


def _split_fused_row(row: List[str], known_names: set) -> List[List[str]]:
    """Split fused rows from MinerU output."""
    if not row:
        return [row]
    first = _clean_cell(row[0])
    tokens = first.split()
    valid = [t for t in tokens if t in known_names or _ALLOY_NAME_RE.match(t)]
    if len(valid) < 2:
        return [row]
    n = len(valid)
    out = [[] for _ in range(n)]
    for cell in row:
        parts = _clean_cell(cell).split()
        parts = parts + [""] * (n - len(parts)) if len(parts) < n else parts[:n]
        for i in range(n):
            out[i].append(parts[i])
    for i in range(n):
        out[i][0] = valid[i]
    return out


def _clean_n_phase_remark(s: str) -> Optional[str]:
    """Clean n-phase remarks."""
    s = _clean_cell(s)
    if not s:
        return None
    low = s.lower()
    if low.startswith(("from ", "phase ", "stable ", "aging ")):
        return None
    if s.endswith(","):
        return None
    half = len(s) // 2
    if half > 4 and s[:half].strip() == s[half:].strip():
        s = s[:half].strip()
    tokens = s.split()
    if len(tokens) >= 4 and tokens[: len(tokens) // 2] == tokens[len(tokens) // 2:]:
        s = " ".join(tokens[: len(tokens) // 2])
    return s or None


def _build_solvus(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build solvus data from tables."""
    known_names: set = set()
    for rows in _find_composition_tables(items):
        for r in rows:
            if r:
                nm = _clean_cell(r[0])
                if nm and _ALLOY_NAME_RE.match(nm):
                    known_names.add(nm)

    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if it.get("type") != "table":
            continue
        rows = _parse_html_table(it.get("table_body", ""))
        header_idx = None
        for i, r in enumerate(rows):
            joined_lower = " ".join(_clean_cell(c).lower() for c in r)
            joined_orig = " ".join(r)
            if "10hv" in joined_lower and ("δ-solv" in joined_orig.lower() or "γ" in joined_orig or "d-solv" in joined_lower):
                header_idx = i
                break
        if header_idx is None:
            continue

        for r in rows[header_idx + 1:]:
            if not r or not any(_clean_cell(c) for c in r):
                continue
            for sub in _split_fused_row(r, known_names):
                name = _clean_cell(sub[0])
                if not name or not _ALLOY_NAME_RE.match(name):
                    continue
                def g(i):
                    return _clean_cell(sub[i]) if i < len(sub) else ""
                remark = g(5) or None
                if remark:
                    remark = _clean_n_phase_remark(remark)
                entry = {
                    "delta_solvus_C": _parse_numeric(g(1)),
                    "gamma_prime_solvus_C": _parse_numeric(g(2)),
                    "delta_T_K": _parse_numeric(g(3)),
                    "hardness_10HV": _parse_numeric(g(4)),
                    "n_phase_remark": remark,
                }
                prev = out.get(name)
                if prev and sum(v is not None for v in entry.values()) < sum(v is not None for v in prev.values()):
                    continue
                out[name] = entry
    return out


def _build_mechanical_a780(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build mechanical properties from Table 8."""
    rows = _find_mechanical_a780_table(items)
    if not rows:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    temps = [20, 650, 700, 750]
    cols_per_temp = ["Rp0.2_MPa", "Rm_MPa", "A5_pct", "Z_pct"]
    for r in rows:
        if not r:
            continue
        first = _clean_cell(r[0])
        if not first:
            continue
        if not (first.isdigit() or first.lower().startswith("a718")):
            continue
        vals = [_parse_numeric(c) for c in r[1:]]
        entry: Dict[str, Any] = {}
        for ti, temp in enumerate(temps):
            for ci, col in enumerate(cols_per_temp):
                idx = ti * 4 + ci
                if idx < len(vals):
                    entry[f"{col}_at_{temp}C"] = vals[idx]
        alloy_key = f"Batch {first}" if first.isdigit() else first
        if first.isdigit():
            entry["material_family"] = "A780 (VDM Alloy 780 Premium)"
        else:
            entry["material_family"] = "A718"
        out[alloy_key] = entry
    return out


# ---------------------------------------------------------------------------
# MCP Tools (public API)
# ---------------------------------------------------------------------------

@mcp.tool()
def list_all_patents() -> List[str]:
    """List available patent folder names under ./patents/."""
    if not PATENTS_DIR.exists():
        return []
    return sorted(p.name for p in PATENTS_DIR.iterdir() if p.is_dir())


@mcp.tool()
def get_patent_metadata(patent_id: str) -> Dict[str, Any]:
    """Return cover-page metadata for a patent."""
    folder = _resolve_patent_dir(patent_id)
    if isinstance(folder, dict):
        return folder
    
    items = _load_content_list(folder)
    md_text = _load_markdown(folder)
    md = _extract_metadata(items, md_text)
    md["patent_id"] = patent_id
    return md


@mcp.tool()
def list_alloys(patent_id: str) -> Dict[str, Any]:
    """Return the list of alloys found in the patent and a short summary."""
    folder = _resolve_patent_dir(patent_id)
    if isinstance(folder, dict):
        return folder
    
    items = _load_content_list(folder)
    md_text = _load_markdown(folder)
    
    wt = _build_composition_wt(items, md_text)
    at = _build_composition_at(items)
    solvus = _build_solvus(items)
    mech = _build_mechanical_a780(items)
    
    # Also add mechanical properties from markdown for Russian patents
    if md_text and not mech:
        ru_mech = _parse_mechanical_properties_ru(md_text)
        if ru_mech:
            for alloy_name, props in ru_mech.items():
                mech[alloy_name] = props
    
    alloys = sorted(set(wt.keys()) | set(at.keys()) | set(solvus.keys()) | set(mech.keys()))
    
    return {
        "patent_id": patent_id,
        "alloys": alloys,
        "alloy_count": len(alloys),
        "tables_found": {
            "composition_wt_tables": sum(1 for _ in _find_composition_tables(items)),
            "composition_at": bool(at),
            "solvus": bool(solvus),
            "mechanical_a780": bool(mech),
            "markdown_fallback": bool(md_text and wt and not _find_composition_tables(items))
        },
        "mechanical_batches": sorted(mech.keys()) if mech else [],
    }


@mcp.tool()
def extract_alloy_row(patent_id: str, alloy_name: str) -> Dict[str, Any]:
    """Return a single, structured dataset row for one alloy in a patent."""
    folder = _resolve_patent_dir(patent_id)
    if isinstance(folder, dict):
        return folder
    
    items = _load_content_list(folder)
    md_text = _load_markdown(folder)

    md = _extract_metadata(items, md_text)
    wt_all = _build_composition_wt(items, md_text)
    at_all = _build_composition_at(items)
    solvus_all = _build_solvus(items)
    mech_all = _build_mechanical_a780(items)
    
    # Add Russian mechanical properties if available
    if md_text and not mech_all:
        ru_mech = _parse_mechanical_properties_ru(md_text)
        if ru_mech:
            mech_all.update(ru_mech)

    wt = wt_all.get(alloy_name, {})
    at = at_all.get(alloy_name, {})
    solvus = solvus_all.get(alloy_name, {})
    mech = mech_all.get(alloy_name, {})

    if not (wt or at or solvus or mech):
        return {
            "error": f"Alloy '{alloy_name}' not found in patent '{patent_id}'",
            "available_alloys": sorted(set(wt_all) | set(at_all) | set(solvus_all) | set(mech_all)),
        }

    process = _extract_process_hints(items)

    # classify role
    ref_alloys = {"A718", "A718Plus", "Waspaloy"}
    role = "reference" if alloy_name in ref_alloys else "test" if re.match(r"^[VL]\d+$", alloy_name) else "batch"
    
    if alloy_name.startswith("RU_") or alloy_name in ["proposed_alloy", "prototype"]:
        role = "composition_only"

    row = {
        "patent_id": patent_id,
        "patent_number": md.get("patent_number"),
        "title": md.get("title"),
        "authors": md.get("authors"),
        "applicant": md.get("applicant"),
        "country": md.get("country"),
        "filing_date": md.get("filing_date"),
        "publication_date": md.get("publication_date"),
        "priority_date": md.get("priority_date"),
        "alloy_name": alloy_name,
        "alloy_role": role,
        "material_base": _infer_material_base(wt),
        "melting_type": process.get("melting_type"),
        "size": process.get("size"),
        "composition_wt_pct": wt,
        "composition_at_pct": at,
        "delta_solvus_C": solvus.get("delta_solvus_C"),
        "gamma_prime_solvus_C": solvus.get("gamma_prime_solvus_C"),
        "delta_T_K": solvus.get("delta_T_K"),
        "hardness_10HV": solvus.get("hardness_10HV"),
        "n_phase_remark": solvus.get("n_phase_remark"),
        "mechanical": mech,
    }
    return row


@mcp.tool()
def extract_all_alloy_rows(patent_id: str) -> Dict[str, Any]:
    """Return ALL dataset rows for a patent at once."""
    summary = list_alloys(patent_id)
    if "error" in summary:
        return summary
    rows = []
    for name in summary["alloys"]:
        r = extract_alloy_row(patent_id, name)
        if "error" not in r:
            rows.append(r)
    return {"patent_id": patent_id, "row_count": len(rows), "rows": rows}


# Fixed element order — one column per element, so all patents share the same schema.
_WT_ELEMENTS = ["Ni", "Fe", "Cr", "Mo", "Ti", "Al", "Nb + Ta", "Nb", "Ta",
                "Co", "C", "Mn", "P", "S", "Si", "B", "Cu", "Zr", "W", "V",
                "Mg", "Ca", "N", "O", "Al + Ti", "Ce", "Hf", "Re"]
_AT_KEYS = ["Al", "Ti", "Co", "Al + Ti", "Al/Ti"]
_MECH_TEMPS = [20, 650, 700, 750]
_MECH_COLS = ["Rp0.2_MPa", "Rm_MPa", "A5_pct", "Z_pct"]


def _flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested alloy row into a single-level dict suitable for CSV."""
    flat: Dict[str, Any] = {
        "patent_id": row.get("patent_id"),
        "patent_number": row.get("patent_number"),
        "title": row.get("title"),
        "authors": "; ".join(row.get("authors") or []),
        "applicant": row.get("applicant"),
        "country": row.get("country"),
        "filing_date": row.get("filing_date"),
        "publication_date": row.get("publication_date"),
        "priority_date": row.get("priority_date"),
        "alloy_name": row.get("alloy_name"),
        "alloy_role": row.get("alloy_role"),
        "material_base": row.get("material_base"),
        "melting_type": row.get("melting_type"),
        "size": row.get("size"),
    }
    wt = row.get("composition_wt_pct") or {}
    for el in _WT_ELEMENTS:
        if isinstance(wt.get(el), dict):
            flat[f"wt_{el}"] = wt[el].get('value') if wt[el] else None
        else:
            flat[f"wt_{el}"] = wt.get(el)
    at = row.get("composition_at_pct") or {}
    for k in _AT_KEYS:
        flat[f"at_{k}"] = at.get(k)
    flat["delta_solvus_C"] = row.get("delta_solvus_C")
    flat["gamma_prime_solvus_C"] = row.get("gamma_prime_solvus_C")
    flat["delta_T_K"] = row.get("delta_T_K")
    flat["hardness_10HV"] = row.get("hardness_10HV")
    flat["n_phase_remark"] = row.get("n_phase_remark")
    mech = row.get("mechanical") or {}
    for temp in _MECH_TEMPS:
        for col in _MECH_COLS:
            flat[f"{col}_at_{temp}C"] = mech.get(f"{col}_at_{temp}C")
    flat["mechanical_family"] = mech.get("material_family")
    flat["UTS_MPa"] = mech.get("UTS_MPa")
    flat["YS_MPa"] = mech.get("YS_MPa")
    flat["elongation_pct"] = mech.get("elongation_pct")
    flat["stress_rupture_100h_MPa"] = mech.get("stress_rupture_100h_MPa")
    flat["creep_rate"] = mech.get("creep_rate")
    flat["test_temperature_C"] = mech.get("test_temperature_C")
    return flat


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> Path:
    """Write flattened rows to CSV, auto-creating the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


@mcp.tool()
def save_patent_csv(patent_id: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
    """Extract all alloys from ONE patent and write them to a CSV file on disk."""
    data = extract_all_alloy_rows(patent_id)
    if "error" in data:
        return data
    flat_rows = [_flatten_row(r) for r in data["rows"]]

    fname = output_filename or f"{patent_id}.csv"
    out_path = Path(fname)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = OUTPUT_DIR / fname

    _write_csv(flat_rows, out_path)
    cols = list(flat_rows[0].keys()) if flat_rows else []
    return {
        "csv_path": str(out_path.resolve()),
        "row_count": len(flat_rows),
        "columns_count": len(cols),
        "columns": cols,
    }


@mcp.tool()
def save_all_patents_csv(output_filename: str = "all_patents.csv") -> Dict[str, Any]:
    """Extract alloys from ALL patents in ./patents/ and write a single combined CSV."""
    patents = list_all_patents()
    all_flat: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for pid in patents:
        data = extract_all_alloy_rows(pid)
        if "error" in data:
            errors.append({"patent_id": pid, "error": data["error"]})
            continue
        for r in data["rows"]:
            all_flat.append(_flatten_row(r))

    out_path = Path(output_filename)
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = OUTPUT_DIR / output_filename

    _write_csv(all_flat, out_path)
    return {
        "csv_path": str(out_path.resolve()),
        "row_count": len(all_flat),
        "patent_count": len(patents) - len(errors),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_patent_dir(patent_id: str):
    if not PATENTS_DIR.exists():
        return {"error": f"Directory '{PATENTS_DIR}' not found"}
    for p in PATENTS_DIR.iterdir():
        if p.is_dir() and (p.name == patent_id or patent_id in p.name):
            return p
    return {"error": f"Patent '{patent_id}' not found"}


def _infer_material_base(wt: Dict[str, Any]) -> Optional[str]:
    """Rough classification: nickel-base / iron-base / cobalt-base from wt%."""
    if not wt:
        return None
    # Handle dict values
    if isinstance(wt.get("Ni"), dict):
        ni_val = wt["Ni"].get('value') if wt["Ni"] else None
        co_val = wt.get("Co", {}).get('value') if isinstance(wt.get("Co"), dict) else wt.get("Co")
    else:
        ni_val = wt.get("Ni")
        co_val = wt.get("Co")
    
    if ni_val == "rest" or (isinstance(ni_val, (int, float)) and ni_val > 50):
        return "nickel-base"
    if co_val and isinstance(co_val, (int, float)) and co_val >= 10:
        return "cobalt-base"
    if isinstance(wt.get("Fe"), (int, float)) and wt["Fe"] > 50:
        return "iron-base"
    return None


_MELTING_RE = re.compile(
    r"(VIM\s*/\s*ESR\s*/\s*VAR|VIM/ESR/VAR|VAR|VIM|ESR|vacuum\s+arc|triple-?melt|double-?melt)",
    re.I,
)
_SIZE_RE = re.compile(r"diameter\s+of\s+(\d+\s*mm)", re.I)


def _extract_process_hints(items: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    texts = [it.get("text", "") for it in items if it.get("type") == "text"]
    joined = " ".join(texts)
    melt = None
    m = _MELTING_RE.search(joined)
    if m:
        melt = m.group(1)
    size = None
    m = _SIZE_RE.search(joined)
    if m:
        size = f"diameter {m.group(1)}"
    return {"melting_type": melt, "size": size}


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=3000)