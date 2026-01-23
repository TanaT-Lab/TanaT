#!/usr/bin/env python3
"""
Generate llms.txt and llms-full.txt files for LLM consumption.

This script creates optimized documentation files:
- llms.txt: Structured index with links to documentation (auto-discovered)
- llms-full.txt: Full content optimized for LLMs (minimal tokens, max info)

Usage:
    python scripts/generate_llms_txt.py
    python scripts/generate_llms_txt.py --output-dir public
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Base paths
SCRIPT_DIR = Path(__file__).parent
DOC_SOURCE = SCRIPT_DIR.parent / "doc" / "source"
SRC_DIR = SCRIPT_DIR.parent / "src"

# Import version
sys.path.insert(0, str(SRC_DIR))
from tanat.version import VERSION

# Extract clean version (major.minor.patch only)
VERSION_PARTS = VERSION.split(".")
CLEAN_VERSION = ".".join(VERSION_PARTS[:3]) if len(VERSION_PARTS) >= 3 else VERSION

# Configuration
DOCS_URL = "https://tanat.gitlabpages.inria.fr/core/tanat"
PROJECT_NAME = "TanaT"
PROJECT_SUMMARY = """TanaT (Temporal ANalysis of Trajectories) is a Python library for temporal sequence analysis, focused on patient care pathways. It supports multi-sequence trajectories combining events, intervals, and states."""


# =============================================================================
# Auto-Discovery Configuration
# =============================================================================

# Static pages (manually maintained - small list)
STATIC_PAGES = {
    "Getting Started": [
        ("Installation", "getting-started/installation"),
        ("Core Concepts", "getting-started/concepts"),
        ("First Steps", "getting-started/first-steps"),
    ],
    "Community": [
        ("Changelog", "community/changelog"),
        ("Contributing", "community/contributing"),
        ("Citing TanaT", "community/citing"),
    ],
}

# Auto-scan configuration for index (llms.txt)
INDEX_SCAN_DIRS = {
    "Examples": {
        "path": "user-guide/auto_examples",
        "extensions": [".rst"],
        "exclude": ["index.rst", "sg_execution_times.rst", "README.rst"],
        "url_prefix": "user-guide/auto_examples",
    },
    "Tutorials": {
        "path": "user-guide/tutorials",
        "extensions": [".py"],
        "exclude": ["README.rst"],
        "url_prefix": "user-guide/tutorials",
    },
    "Reference": {
        "path": "reference",
        "extensions": [".rst"],
        "exclude": ["index.rst", "api/", "glossary.rst"],  # Exclude glossary from index
        "url_prefix": "reference",
    },
}

# Auto-scan configuration for full content (llms-full.txt)
# Each section becomes a separate llms/<section>.txt file
CONTENT_SECTIONS = {
    "getting-started": {
        "title": "Getting Started",
        "files": [
            ("getting-started/concepts.rst", "rst"),
            ("getting-started/first-steps.rst", "rst"),
            ("getting-started/installation.rst", "rst"),
        ],
    },
    "reference": {
        "title": "API Reference",
        "path": "reference",
        "extensions": [".rst"],
        "exclude": ["index.rst", "api/", "glossary.rst"],
    },
    "examples": {
        "title": "Examples",
        "path": "user-guide/examples",
        "extensions": [".py"],
        "exclude": ["README.rst"],
    },
    "tutorials": {
        "title": "Tutorials",
        "path": "user-guide/tutorials",
        "extensions": [".py"],
        "exclude": ["README.rst"],
    },
    "community": {
        "title": "Community",
        "files": [
            ("community/contributing.rst", "rst"),
            ("community/citing.rst", "rst"),
        ],
    },
}


# =============================================================================
# Auto-Discovery Functions
# =============================================================================


def extract_title_from_py(content: str) -> Optional[str]:
    """Extract title from Python file docstring or first markdown header."""
    match = re.search(r'^"""[\s\n]*([^\n]+)', content)
    if match:
        title = match.group(1).strip()
        if title and not title.startswith("="):
            return title
    match = re.search(r"^# %% \[markdown\]\s*\n# # ([^\n]+)", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def extract_title_from_rst(content: str) -> Optional[str]:
    """Extract title from RST file."""
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if i > 0 and re.match(r"^[=]+$", line.strip()):
            return lines[i - 1].strip()
        if i < len(lines) - 1 and re.match(r"^[=]+$", lines[i + 1].strip()):
            return line.strip()
    return None


def discover_files(
    config: Dict, for_index: bool = False
) -> List[Tuple[str, str, Optional[str]]]:
    """Discover files based on config. Returns list of (rel_path, file_type, title)."""
    files = []
    section_path = DOC_SOURCE / config["path"]

    if not section_path.exists():
        print(f"Warning: Directory not found: {section_path}")
        return files

    for ext in config["extensions"]:
        for file_path in sorted(section_path.rglob(f"*{ext}")):
            rel_path = file_path.relative_to(DOC_SOURCE)

            # Check exclusions
            skip = False
            for exclude in config.get("exclude", []):
                if exclude in str(rel_path):
                    skip = True
                    break

            if skip:
                continue

            # Extract title
            content = file_path.read_text(encoding="utf-8")
            if ext == ".py":
                title = extract_title_from_py(content)
                file_type = "py"
            else:
                title = extract_title_from_rst(content)
                file_type = "rst"

            if not title:
                title = file_path.stem.replace("_", " ").replace("-", " ").title()

            # For index, use URL path (title, url_path) to match STATIC_PAGES format
            if for_index:
                url_path = str(rel_path).replace(".rst", "").replace(".py", "")
                files.append((title, url_path))
            else:
                files.append((str(rel_path), file_type, title))

    return files


def discover_docs_structure() -> Dict[str, List[Tuple[str, str]]]:
    """Auto-discover documentation structure for index."""
    structure = {}

    # Add static pages first
    structure.update(STATIC_PAGES)

    # Auto-discover other sections
    for section_name, config in INDEX_SCAN_DIRS.items():
        files = discover_files(config, for_index=True)
        if files:
            structure[section_name] = files

    return structure


def discover_content_files() -> List[Tuple[str, str]]:
    """Discover all content files for llms-full.txt (legacy, for compatibility)."""
    files = []

    for section_name, config in CONTENT_SECTIONS.items():
        section_files = discover_section_files(section_name)
        files.extend(section_files)

    return files


def discover_section_files(section_name: str) -> List[Tuple[str, str]]:
    """Discover files for a specific section."""
    config = CONTENT_SECTIONS.get(section_name, {})

    # If explicit file list provided
    if "files" in config:
        return config["files"]

    # Otherwise auto-discover
    files = []
    if "path" in config:
        section_path = DOC_SOURCE / config["path"]

        if not section_path.exists():
            print(f"Warning: Directory not found: {section_path}")
            return files

        for ext in config.get("extensions", []):
            for file_path in sorted(section_path.rglob(f"*{ext}")):
                rel_path = file_path.relative_to(DOC_SOURCE)

                # Check exclusions
                skip = False
                for exclude in config.get("exclude", []):
                    if exclude in str(rel_path):
                        skip = True
                        break

                if skip:
                    continue

                file_type = "py" if ext == ".py" else "rst"
                files.append((str(rel_path), file_type))

    return files


# =============================================================================
# Content Cleaning Functions
# =============================================================================


def remove_llm_ignore_sections(content: str) -> str:
    """Remove sections marked with .. llm-ignore:: directive."""
    # Pattern: .. llm-ignore:: until .. llm-end-ignore::
    pattern = r"\.\. llm-ignore::.*?\.\. llm-end-ignore::"
    content = re.sub(pattern, "", content, flags=re.DOTALL)
    return content


def clean_rst_content(content: str) -> str:
    """Clean RST content for LLM consumption."""
    # First, remove llm-ignore sections
    content = remove_llm_ignore_sections(content)

    lines = content.split("\n")
    cleaned_lines = []
    skip_until_blank = False

    for line in lines:
        # Skip directive blocks (keep topic, note, code-block which contain useful content)
        if line.strip().startswith(".. ") and "::" in line:
            directive = line.strip().split("::")[0].replace(".. ", "")
            if directive in [
                "toctree",
                "mermaid",
                "image",
                "figure",
                "warning",
                "seealso",
                "raw",
                "only",
                "container",
                "tip",
                "hint",
                "important",
                "attention",
            ]:
                skip_until_blank = True
                continue
            # For code-block: skip the directive line but keep content
            if directive == "code-block":
                continue  # Skip ".. code-block:: xxx" line, keep indented content
            # Keep topic and note as-is for consistency with source docs

        if skip_until_blank:
            # Continue skipping until we hit a non-indented, non-empty line
            if line.strip() == "":
                continue
            if line and line[0].isspace():
                continue
            skip_until_blank = False

        # Skip RST-specific lines (but keep table content)
        if line.strip().startswith(":") and ":" in line.strip()[1:]:
            parts = line.strip().split(":")
            if len(parts) > 1 and parts[1] in ["maxdepth", "caption", "hidden"]:
                continue
        if line.strip().startswith(".. _"):  # Labels
            continue
        if re.match(r'^[=\-~`\'"^]+$', line.strip()) and len(line.strip()) > 3:
            continue  # Underlines (keep headers)

        # Skip orphan arrows (residual from cleaned links)
        if line.strip() in ["→", "←", "↔", "->"]:
            continue

        # Skip lines that are just whitespace + dashes (table separators) or RST separators
        if re.match(r"^\s*[\-\+\|]+\s*$", line):
            continue

        # Skip RST horizontal rules (----)
        if re.match(r"^-{4,}$", line.strip()):
            continue

        # Clean inline markup - RST links
        line = re.sub(r"`([^`<]+)\s*<[^>]+>`_", r"\1", line)  # `Text <url>`_ -> Text
        line = re.sub(r":doc:`[^`]*`", "", line)  # Remove :doc: refs
        line = re.sub(r":ref:`([^`]*)`", r"\1", line)  # Simplify :ref:
        line = re.sub(r":class:`~?([^`]*)`", r"`\1`", line)  # Simplify :class:
        line = re.sub(r":meth:`~?([^`]*)`", r"`\1`", line)  # Simplify :meth:
        line = re.sub(r":func:`~?([^`]*)`", r"`\1`", line)  # Simplify :func:
        line = re.sub(r":py:[a-z]+:`~?([^`]*)`", r"`\1`", line)  # Simplify :py:*:
        line = re.sub(r":py`([^`]*)`", r"`\1`", line)  # Simplify :py`...`
        line = re.sub(r":attr:`~?([^`]*)`", r"`\1`", line)  # Simplify :attr:

        # Clean RST double backticks to single (``code`` -> `code`)
        line = re.sub(r"``([^`]+)``", r"`\1`", line)

        # Remove standalone arrows
        line = re.sub(r"\s*→\s*$", "", line)
        line = re.sub(r"^\s*→\s*", "", line)

        # Skip if line became empty or just whitespace after cleaning
        if line.strip() == "":
            cleaned_lines.append("")
            continue

        cleaned_lines.append(line)

    # Remove excessive blank lines
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def truncate_large_structures(content: str, max_lines: int = 30) -> str:
    """Truncate large dict/list literals based on line count.

    Simple and robust: if a structure spans more than max_lines, truncate it.
    Works line by line for efficiency.
    Only targets standalone assignments (var = {...}), not function arguments.
    """
    lines = content.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Only match standalone assignments at line start: var = { or var = [
        # Skip function arguments like func(param={...})
        match = re.match(r"^(\s*)(\w+)\s*=\s*([{\[])(.*)$", line)
        if match:
            indent = match.group(1)
            varname = match.group(2)
            open_char = match.group(3)
            rest = match.group(4)
            close_char = "}" if open_char == "{" else "]"

            # Skip already abbreviated structures like {...} or [...]
            if rest.strip().startswith("..."):
                result_lines.append(line)
                i += 1
                continue

            # Check if structure closes on the same line (balanced braces)
            depth = 1
            for char in rest:
                if char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1
            if depth == 0:
                # Closes on same line - keep as is
                result_lines.append(line)
                i += 1
                continue

            # Multi-line structure: find the closing line
            start_line = i
            j = i + 1

            while j < len(lines) and depth > 0:
                for char in lines[j]:
                    if char == open_char:
                        depth += 1
                    elif char == close_char:
                        depth -= 1
                j += 1

            num_lines = j - start_line

            if num_lines > max_lines:
                # Truncate
                result_lines.append(
                    f"{indent}{varname} = {{...}}  # {num_lines} lines, truncated for brevity"
                )
                i = j
            else:
                # Keep all lines
                result_lines.extend(lines[start_line:j])
                i = j
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


def clean_py_content(content: str) -> str:
    """Clean Python notebook-style content for LLM consumption."""
    # Truncate large data structures (dicts/lists > 8 lines)
    content = truncate_large_structures(content)

    lines = content.split("\n")
    cleaned_lines = []
    in_markdown_block = False

    for line in lines:
        # Convert markdown cell markers
        if line.strip() == "# %% [markdown]":
            in_markdown_block = True
            continue

        if line.strip() == "# %%":
            in_markdown_block = False
            cleaned_lines.append("")  # Blank line before code
            continue

        if in_markdown_block:
            # Remove leading '# ' from markdown
            if line.startswith("# "):
                cleaned_lines.append(line[2:])
            elif line.strip() == "#":
                cleaned_lines.append("")
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    # Remove excessive blank lines
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


# =============================================================================
# Generation Functions
# =============================================================================


def generate_llms_txt(section_stats: Dict[str, int]) -> str:
    """Generate the llms.txt index file.

    Creates a clean text-only index pointing to section files.
    """
    lines = [
        f"# {PROJECT_NAME}",
        "",
        f"> {PROJECT_SUMMARY}",
        f"> [HTML documentation]({DOCS_URL})",
        f"> Version: {CLEAN_VERSION}",
        "",
        "## Documentation Files",
        "",
        "Documentation is available as plain text files optimized for LLMs:",
        "",
    ]

    # List section files with Markdown URLs and token estimates
    for section_name, config in CONTENT_SECTIONS.items():
        title = config.get("title", section_name.replace("-", " ").title())
        tokens = section_stats.get(section_name, 0)
        url = f"{DOCS_URL}/llms/{section_name}.txt"
        lines.append(f"- [{title}]({url}) (~{tokens:,} tokens)")

    lines.append("")
    full_url = f"{DOCS_URL}/llms-full.txt"
    lines.append(
        f"- [All content combined]({full_url}) (~{sum(section_stats.values()):,} tokens)"
    )
    lines.append("")

    # Table of contents
    lines.append("## Table of Contents")
    lines.append("")

    # Get auto-discovered docs structure for TOC
    docs_structure = discover_docs_structure()

    for section, entries in docs_structure.items():
        lines.append(f"### {section}")
        lines.append("")
        for title, path in entries:
            lines.append(f"- {title}")
        lines.append("")

    return "\n".join(lines)


def generate_section_txt(section_name: str) -> str:
    """Generate a single section file."""
    config = CONTENT_SECTIONS.get(section_name, {})
    title = config.get("title", section_name.replace("-", " ").title())

    lines = [
        f"# {PROJECT_NAME} - {title}",
        "",
        "=" * 60,
        "",
    ]

    files = discover_section_files(section_name)

    for rel_path, file_type in files:
        file_path = DOC_SOURCE / rel_path

        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        content = file_path.read_text(encoding="utf-8")

        # Extract title and clean content
        if file_type == "py":
            file_title = extract_title_from_py(content)
            cleaned = clean_py_content(content)
        else:
            file_title = extract_title_from_rst(content)
            cleaned = clean_rst_content(content)

        if not file_title:
            file_title = file_path.stem.replace("_", " ").title()

        # Add file content
        lines.append(f"## {file_title}")
        lines.append("")
        lines.append(cleaned)
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    return "\n".join(lines)


def generate_llms_full_txt() -> str:
    """Generate the llms-full.txt by concatenating all sections."""
    sections = [
        f"# {PROJECT_NAME} - Complete Documentation",
        "",
        PROJECT_SUMMARY,
        "",
        "=" * 60,
        "",
    ]

    for section_name in CONTENT_SECTIONS.keys():
        section_content = generate_section_txt(section_name)
        # Remove the header from section (already have main header)
        lines = section_content.split("\n")
        # Skip first 4 lines (header + separator)
        content_start = 0
        for i, line in enumerate(lines):
            if line.startswith("## "):
                content_start = i
                break
        sections.append("\n".join(lines[content_start:]))
        sections.append("")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Generate llms.txt files for LLM consumption"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="public",
        help="Output directory (default: public)",
    )
    args = parser.parse_args()

    # Resolve output directory relative to tanat root
    output_dir = SCRIPT_DIR.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create llms subdirectory
    llms_dir = output_dir / "llms"
    llms_dir.mkdir(parents=True, exist_ok=True)

    # Generate section files
    section_stats = {}
    total_files = 0

    for section_name in CONTENT_SECTIONS.keys():
        section_content = generate_section_txt(section_name)
        section_path = llms_dir / f"{section_name}.txt"
        section_path.write_text(section_content, encoding="utf-8")

        tokens = len(section_content) // 4
        section_stats[section_name] = tokens
        files_count = len(discover_section_files(section_name))
        total_files += files_count

        print(
            f"Generated: llms/{section_name}.txt ({files_count} files, ~{tokens:,} tokens)"
        )

    # Generate llms-full.txt (concatenation of all sections)
    llms_full_txt = generate_llms_full_txt()
    llms_full_path = output_dir / "llms-full.txt"
    llms_full_path.write_text(llms_full_txt, encoding="utf-8")
    print(
        f"Generated: llms-full.txt ({total_files} files, ~{len(llms_full_txt)//4:,} tokens)"
    )

    # Generate llms.txt index (with section stats)
    llms_txt = generate_llms_txt(section_stats)
    llms_path = output_dir / "llms.txt"
    llms_path.write_text(llms_txt, encoding="utf-8")
    print(f"Generated: llms.txt (index, ~{len(llms_txt)//4:,} tokens)")

    # Summary
    print(f"\nTotal: {total_files} documentation files")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
