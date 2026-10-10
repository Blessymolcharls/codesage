"""Syntax-aware and fallback code unit extractors for CodeSage."""

import ast
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CodeUnit:
    """Represents an extracted code unit."""

    unit_id: str
    repository_relative_path: str
    language: str
    unit_type: str  # function, method, class, module, line_chunk
    symbol_name: Optional[str]
    parent_symbol: Optional[str]
    start_line: int  # 1-based inclusive
    end_line: int  # 1-based inclusive
    source_text: str
    content_sha256: str
    extraction_method: str  # tree_sitter, python_ast, line_chunk
    parser_status: str  # success, partial, fallback
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict representation matching CodeSage schema."""
        return {
            "unit_id": self.unit_id,
            "repository_relative_path": self.repository_relative_path,
            "language": self.language,
            "unit_type": self.unit_type,
            "symbol_name": self.symbol_name,
            "parent_symbol": self.parent_symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "source_text": self.source_text,
            "content_sha256": self.content_sha256,
            "extraction_method": self.extraction_method,
            "parser_status": self.parser_status,
            "warnings": self.warnings,
        }


def make_unit_id(
    rel_path: str,
    unit_type: str,
    symbol_name: Optional[str],
    start_line: int,
    end_line: int,
    part_idx: Optional[int] = None,
) -> str:
    """Generate a stable, deterministic identifier for a code unit."""
    name_str = symbol_name if symbol_name else "anon"
    base = f"{rel_path}:{unit_type}:{name_str}:{start_line}_{end_line}"
    if part_idx is not None:
        return f"{base}:part{part_idx}"
    return base


def compute_sha256(text: str) -> str:
    """Compute sha256 hex digest of string content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_large_unit_if_needed(
    base_unit: CodeUnit,
    lines: List[str],
    max_unit_lines: int,
) -> List[CodeUnit]:
    """Split units exceeding max_unit_lines into deterministic non-overlapping chunks."""
    total_lines = base_unit.end_line - base_unit.start_line + 1
    if total_lines <= max_unit_lines:
        return [base_unit]

    chunks: List[CodeUnit] = []
    current_start = base_unit.start_line
    part_idx = 1

    while current_start <= base_unit.end_line:
        current_end = min(current_start + max_unit_lines - 1, base_unit.end_line)
        chunk_lines = lines[current_start - 1 : current_end]
        chunk_text = "".join(chunk_lines)

        chunk_id = make_unit_id(
            rel_path=base_unit.repository_relative_path,
            unit_type=base_unit.unit_type,
            symbol_name=base_unit.symbol_name,
            start_line=current_start,
            end_line=current_end,
            part_idx=part_idx,
        )

        warning_msg = (
            f"Split large {base_unit.unit_type} '{base_unit.symbol_name or 'anon'}' "
            f"({total_lines} lines > {max_unit_lines}) into part {part_idx} "
            f"(lines {current_start}-{current_end})"
        )

        chunk_unit = CodeUnit(
            unit_id=chunk_id,
            repository_relative_path=base_unit.repository_relative_path,
            language=base_unit.language,
            unit_type=base_unit.unit_type,
            symbol_name=base_unit.symbol_name,
            parent_symbol=base_unit.parent_symbol,
            start_line=current_start,
            end_line=current_end,
            source_text=chunk_text,
            content_sha256=compute_sha256(chunk_text),
            extraction_method=base_unit.extraction_method,
            parser_status=base_unit.parser_status,
            warnings=list(base_unit.warnings) + [warning_msg],
        )
        chunks.append(chunk_unit)
        current_start = current_end + 1
        part_idx += 1

    return chunks


class LineChunkExtractor:
    """Deterministic line-based fallback chunker."""

    def __init__(self, chunk_size: int = 50):
        self.chunk_size = chunk_size

    def extract(
        self,
        rel_path: str,
        source_text: str,
        language: str,
        reason: str = "Fallback line chunk extraction",
    ) -> List[CodeUnit]:
        lines = source_text.splitlines(keepends=True)
        if not lines:
            return []

        units: List[CodeUnit] = []
        total_lines = len(lines)
        chunk_start = 1

        while chunk_start <= total_lines:
            chunk_end = min(chunk_start + self.chunk_size - 1, total_lines)
            chunk_slice = lines[chunk_start - 1 : chunk_end]
            chunk_text = "".join(chunk_slice)

            unit_id = make_unit_id(
                rel_path=rel_path,
                unit_type="line_chunk",
                symbol_name=None,
                start_line=chunk_start,
                end_line=chunk_end,
            )

            units.append(
                CodeUnit(
                    unit_id=unit_id,
                    repository_relative_path=rel_path,
                    language=language,
                    unit_type="line_chunk",
                    symbol_name=None,
                    parent_symbol=None,
                    start_line=chunk_start,
                    end_line=chunk_end,
                    source_text=chunk_text,
                    content_sha256=compute_sha256(chunk_text),
                    extraction_method="line_chunk",
                    parser_status="fallback",
                    warnings=[reason],
                )
            )
            chunk_start = chunk_end + 1

        return units


class PythonAstExtractor:
    """Extract functions, methods, and classes using Python's built-in AST."""

    def extract(
        self,
        rel_path: str,
        source_text: str,
        max_unit_lines: int = 250,
    ) -> Tuple[List[CodeUnit], str, List[str]]:
        lines = source_text.splitlines(keepends=True)
        warnings: List[str] = []

        try:
            tree = ast.parse(source_text, filename=rel_path)
        except SyntaxError as e:
            return (
                [],
                "fallback",
                [f"Python syntax error at line {e.lineno}: {e.msg}"],
            )

        raw_units: List[CodeUnit] = []

        def get_slice(start_l: int, end_l: int) -> str:
            return "".join(lines[start_l - 1 : end_l])

        def visit_node(node: ast.AST, parent_class: Optional[str] = None):
            if isinstance(node, ast.ClassDef):
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                unit_text = get_slice(start, end)
                raw_units.append(
                    CodeUnit(
                        unit_id=make_unit_id(rel_path, "class", node.name, start, end),
                        repository_relative_path=rel_path,
                        language="python",
                        unit_type="class",
                        symbol_name=node.name,
                        parent_symbol=parent_class,
                        start_line=start,
                        end_line=end,
                        source_text=unit_text,
                        content_sha256=compute_sha256(unit_text),
                        extraction_method="python_ast",
                        parser_status="success",
                    )
                )
                for item in node.body:
                    visit_node(item, parent_class=node.name)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                unit_text = get_slice(start, end)
                u_type = "method" if parent_class else "function"
                raw_units.append(
                    CodeUnit(
                        unit_id=make_unit_id(rel_path, u_type, node.name, start, end),
                        repository_relative_path=rel_path,
                        language="python",
                        unit_type=u_type,
                        symbol_name=node.name,
                        parent_symbol=parent_class,
                        start_line=start,
                        end_line=end,
                        source_text=unit_text,
                        content_sha256=compute_sha256(unit_text),
                        extraction_method="python_ast",
                        parser_status="success",
                    )
                )
                # Inner functions
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        visit_node(item, parent_class=None)

            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        visit_node(child, parent_class=parent_class)

        for stmt in tree.body:
            visit_node(stmt, parent_class=None)

        # Apply deterministic unit splitting for oversized units
        final_units: List[CodeUnit] = []
        for u in raw_units:
            final_units.extend(split_large_unit_if_needed(u, lines, max_unit_lines))

        return final_units, "success", warnings


class TreeSitterExtractor:
    """Extract code units using Tree-sitter for JavaScript and Dart."""

    def __init__(self):
        self._parsers: Dict[str, Any] = {}
        self._languages: Dict[str, Any] = {}
        self._init_languages()

    def _init_languages(self) -> None:
        try:
            import tree_sitter
            import tree_sitter_javascript
            js_lang = tree_sitter.Language(tree_sitter_javascript.language())
            self._languages["javascript"] = js_lang
            self._parsers["javascript"] = tree_sitter.Parser(js_lang)
        except Exception:
            pass

        try:
            import tree_sitter
            import tree_sitter_dart
            dart_lang = tree_sitter.Language(tree_sitter_dart.language())
            self._languages["dart"] = dart_lang
            self._parsers["dart"] = tree_sitter.Parser(dart_lang)
        except Exception:
            pass

    def is_available(self, language: str) -> bool:
        return language in self._parsers

    def extract(
        self,
        rel_path: str,
        source_text: str,
        language: str,
        max_unit_lines: int = 250,
    ) -> Tuple[List[CodeUnit], str, List[str]]:
        if not self.is_available(language):
            return [], "fallback", [f"Tree-sitter parser unavailable for {language}"]

        parser = self._parsers[language]
        source_bytes = source_text.encode("utf-8")
        try:
            tree = parser.parse(source_bytes)
        except Exception as e:
            return [], "fallback", [f"Tree-sitter parse exception: {e}"]

        lines = source_text.splitlines(keepends=True)
        raw_units: List[CodeUnit] = []
        warnings: List[str] = []

        if language == "javascript":
            raw_units = self._extract_javascript(rel_path, tree.root_node, lines, source_bytes)
        elif language == "dart":
            raw_units = self._extract_dart(rel_path, tree.root_node, lines, source_bytes)

        if not raw_units and len(lines) > 5:
            # File had substantial content but no recognized symbols
            warnings.append(f"Tree-sitter parsed {language} file with no top-level symbols extracted")
            status = "partial"
        else:
            status = "success"

        # Apply large unit splitting deterministically
        final_units: List[CodeUnit] = []
        for u in raw_units:
            final_units.extend(split_large_unit_if_needed(u, lines, max_unit_lines))

        return final_units, status, warnings

    def _get_node_lines(self, node: Any, total_lines: int) -> Tuple[int, int]:
        """Convert tree-sitter start and end points to 1-based inclusive lines within file bounds."""
        s_row = node.start_point[0]
        e_row = node.end_point[0]
        e_col = node.end_point[1]
        start_line = s_row + 1
        end_line = e_row + 1
        if e_col == 0 and e_row > s_row:
            end_line = e_row
        start_line = max(1, min(start_line, total_lines))
        end_line = max(start_line, min(end_line, total_lines))
        return start_line, end_line

    def _extract_javascript(
        self,
        rel_path: str,
        root_node: Any,
        lines: List[str],
        source_bytes: bytes,
    ) -> List[CodeUnit]:
        units: List[CodeUnit] = []
        total_lines = len(lines)

        def get_slice(start_l: int, end_l: int) -> str:
            return "".join(lines[start_l - 1 : end_l])

        def visit(node: Any, parent_class: Optional[str] = None):
            node_type = node.type

            # Class declaration
            if node_type in ("class_declaration", "class"):
                name_node = node.child_by_field_name("name")
                name_str = name_node.text.decode("utf-8", errors="replace") if name_node else None
                s_line, e_line = self._get_node_lines(node, total_lines)
                text = get_slice(s_line, e_line)
                units.append(
                    CodeUnit(
                        unit_id=make_unit_id(rel_path, "class", name_str, s_line, e_line),
                        repository_relative_path=rel_path,
                        language="javascript",
                        unit_type="class",
                        symbol_name=name_str,
                        parent_symbol=parent_class,
                        start_line=s_line,
                        end_line=e_line,
                        source_text=text,
                        content_sha256=compute_sha256(text),
                        extraction_method="tree_sitter",
                        parser_status="success",
                    )
                )

                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        visit(child, parent_class=name_str)
                return

            # Method definition inside class or object
            if node_type == "method_definition":
                name_node = node.child_by_field_name("name")
                name_str = name_node.text.decode("utf-8", errors="replace") if name_node else None
                s_line, e_line = self._get_node_lines(node, total_lines)
                text = get_slice(s_line, e_line)
                units.append(
                    CodeUnit(
                        unit_id=make_unit_id(rel_path, "method", name_str, s_line, e_line),
                        repository_relative_path=rel_path,
                        language="javascript",
                        unit_type="method",
                        symbol_name=name_str,
                        parent_symbol=parent_class,
                        start_line=s_line,
                        end_line=e_line,
                        source_text=text,
                        content_sha256=compute_sha256(text),
                        extraction_method="tree_sitter",
                        parser_status="success",
                    )
                )
                return

            # Function declaration: function foo() {}
            if node_type == "function_declaration":
                name_node = node.child_by_field_name("name")
                name_str = name_node.text.decode("utf-8", errors="replace") if name_node else None
                s_line, e_line = self._get_node_lines(node, total_lines)
                text = get_slice(s_line, e_line)
                units.append(
                    CodeUnit(
                        unit_id=make_unit_id(rel_path, "function", name_str, s_line, e_line),
                        repository_relative_path=rel_path,
                        language="javascript",
                        unit_type="function",
                        symbol_name=name_str,
                        parent_symbol=parent_class,
                        start_line=s_line,
                        end_line=e_line,
                        source_text=text,
                        content_sha256=compute_sha256(text),
                        extraction_method="tree_sitter",
                        parser_status="success",
                    )
                )
                return

            # Lexical / Variable declaration: const foo = () => {} or let foo = function() {}
            if node_type in ("lexical_declaration", "variable_declaration"):
                for child in node.children:
                    if child.type == "variable_declarator":
                        val = child.child_by_field_name("value")
                        if val and val.type in ("arrow_function", "function_expression"):
                            name_node = child.child_by_field_name("name")
                            name_str = name_node.text.decode("utf-8", errors="replace") if name_node else None
                            s_line, e_line = self._get_node_lines(node, total_lines)
                            text = get_slice(s_line, e_line)
                            units.append(
                                CodeUnit(
                                    unit_id=make_unit_id(rel_path, "function", name_str, s_line, e_line),
                                    repository_relative_path=rel_path,
                                    language="javascript",
                                    unit_type="function",
                                    symbol_name=name_str,
                                    parent_symbol=parent_class,
                                    start_line=s_line,
                                    end_line=e_line,
                                    source_text=text,
                                    content_sha256=compute_sha256(text),
                                    extraction_method="tree_sitter",
                                    parser_status="success",
                                )
                            )
                return

            # Traverse child nodes
            for child in node.children:
                visit(child, parent_class=parent_class)

        for child in root_node.children:
            visit(child, parent_class=None)

        return units

    def _extract_dart(
        self,
        rel_path: str,
        root_node: Any,
        lines: List[str],
        source_bytes: bytes,
    ) -> List[CodeUnit]:
        units: List[CodeUnit] = []
        total_lines = len(lines)

        def get_slice(start_l: int, end_l: int) -> str:
            return "".join(lines[start_l - 1 : end_l])

        def find_identifier_in(node: Any) -> Optional[str]:
            if node.type == "identifier":
                return node.text.decode("utf-8", errors="replace")
            for c in node.children:
                res = find_identifier_in(c)
                if res:
                    return res
            return None

        def extract_from_children(children_list: List[Any], parent_class: Optional[str] = None):
            i = 0
            n = len(children_list)
            while i < n:
                child = children_list[i]
                c_type = child.type

                # Class / Enum / Mixin / Extension definitions
                if c_type in ("class_definition", "enum_declaration", "mixin_declaration", "extension_declaration"):
                    name_str = None
                    id_node = child.child_by_field_name("name")
                    if not id_node:
                        for sub in child.children:
                            if sub.type == "identifier":
                                name_str = sub.text.decode("utf-8", errors="replace")
                                break
                    else:
                        name_str = id_node.text.decode("utf-8", errors="replace")

                    s_line, e_line = self._get_node_lines(child, total_lines)
                    text = get_slice(s_line, e_line)
                    units.append(
                        CodeUnit(
                            unit_id=make_unit_id(rel_path, "class", name_str, s_line, e_line),
                            repository_relative_path=rel_path,
                            language="dart",
                            unit_type="class",
                            symbol_name=name_str,
                            parent_symbol=parent_class,
                            start_line=s_line,
                            end_line=e_line,
                            source_text=text,
                            content_sha256=compute_sha256(text),
                            extraction_method="tree_sitter",
                            parser_status="success",
                        )
                    )

                    body_node = None
                    for sub in child.children:
                        if sub.type in ("class_body", "enum_body"):
                            body_node = sub
                            break
                    if body_node:
                        extract_from_children(body_node.children, parent_class=name_str)

                    i += 1
                    continue

                # Method signature in class body
                if c_type == "method_signature":
                    name_str = find_identifier_in(child)
                    start_node = child
                    end_node = child

                    # Check if next child is function_body
                    if i + 1 < n and children_list[i + 1].type == "function_body":
                        end_node = children_list[i + 1]
                        i += 1

                    s_line = start_node.start_point[0] + 1
                    e_line = end_node.end_point[0] + 1
                    if end_node.end_point[1] == 0 and end_node.end_point[0] > start_node.start_point[0]:
                        e_line = end_node.end_point[0]
                    s_line = max(1, min(s_line, total_lines))
                    e_line = max(s_line, min(e_line, total_lines))

                    text = get_slice(s_line, e_line)
                    units.append(
                        CodeUnit(
                            unit_id=make_unit_id(rel_path, "method", name_str, s_line, e_line),
                            repository_relative_path=rel_path,
                            language="dart",
                            unit_type="method",
                            symbol_name=name_str,
                            parent_symbol=parent_class,
                            start_line=s_line,
                            end_line=e_line,
                            source_text=text,
                            content_sha256=compute_sha256(text),
                            extraction_method="tree_sitter",
                            parser_status="success",
                        )
                    )
                    i += 1
                    continue

                # Top-level function signature
                if c_type == "function_signature":
                    name_str = find_identifier_in(child)
                    start_node = child
                    end_node = child

                    if i + 1 < n and children_list[i + 1].type == "function_body":
                        end_node = children_list[i + 1]
                        i += 1

                    s_line = start_node.start_point[0] + 1
                    e_line = end_node.end_point[0] + 1
                    if end_node.end_point[1] == 0 and end_node.end_point[0] > start_node.start_point[0]:
                        e_line = end_node.end_point[0]
                    s_line = max(1, min(s_line, total_lines))
                    e_line = max(s_line, min(e_line, total_lines))

                    text = get_slice(s_line, e_line)
                    units.append(
                        CodeUnit(
                            unit_id=make_unit_id(rel_path, "function", name_str, s_line, e_line),
                            repository_relative_path=rel_path,
                            language="dart",
                            unit_type="function",
                            symbol_name=name_str,
                            parent_symbol=parent_class,
                            start_line=s_line,
                            end_line=e_line,
                            source_text=text,
                            content_sha256=compute_sha256(text),
                            extraction_method="tree_sitter",
                            parser_status="success",
                        )
                    )
                    i += 1
                    continue

                i += 1

        extract_from_children(root_node.children, parent_class=None)
        return units


class CodeUnitExtractor:
    """Master extractor delegating to language-aware extractors or fallback."""

    def __init__(self, chunk_size: int = 50):
        self.tree_sitter = TreeSitterExtractor()
        self.python_ast = PythonAstExtractor()
        self.line_chunker = LineChunkExtractor(chunk_size=chunk_size)

    def detect_language(self, path: Path) -> str:
        """Infer canonical language name from file extension."""
        ext = path.suffix.lower()
        if ext == ".dart":
            return "dart"
        if ext in (".js", ".jsx", ".mjs", ".cjs"):
            return "javascript"
        if ext in (".ts", ".tsx"):
            return "typescript"
        if ext == ".py":
            return "python"
        return "text"

    def extract_from_file(
        self,
        repo_root: Path,
        file_path: Path,
        max_unit_lines: int = 250,
    ) -> Tuple[List[CodeUnit], str, List[str]]:
        """Extract code units from a repository source file safely."""
        try:
            rel_path = file_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            rel_path = file_path.name

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_text = f.read()
        except UnicodeDecodeError:
            # Fallback to UTF-8 with replace
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    source_text = f.read()
            except Exception as e:
                return (
                    [],
                    "fallback",
                    [f"Failed to read file {rel_path}: {e}"],
                )
        except Exception as e:
            return (
                [],
                "fallback",
                [f"Failed to open file {rel_path}: {e}"],
            )

        if not source_text.strip():
            return [], "success", []

        language = self.detect_language(file_path)

        # 1. Python AST
        if language == "python":
            units, status, warnings = self.python_ast.extract(
                rel_path=rel_path,
                source_text=source_text,
                max_unit_lines=max_unit_lines,
            )
            if status != "fallback" and units:
                return units, status, warnings
            # Fallback to line chunks if AST failed
            fallback_units = self.line_chunker.extract(
                rel_path=rel_path,
                source_text=source_text,
                language=language,
                reason=warnings[0] if warnings else "Python AST fallback",
            )
            return fallback_units, "fallback", warnings

        # 2. Tree-sitter for Dart and JavaScript
        if language in ("javascript", "dart") and self.tree_sitter.is_available(language):
            units, status, warnings = self.tree_sitter.extract(
                rel_path=rel_path,
                source_text=source_text,
                language=language,
                max_unit_lines=max_unit_lines,
            )
            if status != "fallback" and units:
                return units, status, warnings
            # If no units extracted but file is non-empty, use fallback line chunks
            fallback_units = self.line_chunker.extract(
                rel_path=rel_path,
                source_text=source_text,
                language=language,
                reason="Tree-sitter extracted 0 symbols; using line chunk fallback",
            )
            return fallback_units, "fallback", warnings

        # 3. Fallback chunker for other languages or missing parsers
        reason = f"Language '{language}' parser unavailable; using deterministic line chunking"
        fallback_units = self.line_chunker.extract(
            rel_path=rel_path,
            source_text=source_text,
            language=language,
            reason=reason,
        )
        return fallback_units, "fallback", [reason]
