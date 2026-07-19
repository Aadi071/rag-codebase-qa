"""AST-aware chunking: split code on function/class boundaries, not line counts.

Why not just cut every N lines? Because that slices functions in half and
separates code from the context that gives it meaning. Instead we parse the
file into an Abstract Syntax Tree (AST) with tree-sitter, a real parser that
understands the language grammar, and split along the tree's own structure.

The strategy (a small "divide and combine"):
  - Walk the top-level nodes of the file.
  - Greedily pack small adjacent nodes together until they would exceed the
    size budget, then start a new chunk.
  - A function is kept whole even when oversized (splitting its signature from
    its body destroys the semantic unit).
  - A "container" too big to keep whole (a class / Java interface / Rust impl)
    is split into its members, and each member chunk is PREFIXED with the
    container signature and named "Container.member" so it keeps its context.

Every chunk carries metadata (file path, line range, symbol name) which becomes
the auth.py:42 style citation later.
"""

from dataclasses import dataclass

import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_go
import tree_sitter_java
import tree_sitter_rust
from tree_sitter import Language, Parser

# Build each Language + Parser once at import time and reuse for every file.
_LANGUAGES = {
    "python": Language(tree_sitter_python.language()),
    "javascript": Language(tree_sitter_javascript.language()),
    "typescript": Language(tree_sitter_typescript.language_typescript()),
    "tsx": Language(tree_sitter_typescript.language_tsx()),
    "go": Language(tree_sitter_go.language()),
    "java": Language(tree_sitter_java.language()),
    "rust": Language(tree_sitter_rust.language()),
}
_PARSERS = {name: Parser(lang) for name, lang in _LANGUAGES.items()}

# Max characters per chunk. A whole function/class smaller than this stays
# intact; anything larger is broken along its own inner boundaries.
DEFAULT_MAX_CHARS = 1500

# Per-language "container" node types: when oversized, these are split into
# members (with the container signature prepended). Everything else oversized is
# a function (kept whole) or a plain node. Go has no method-holding container
# (methods are top-level funcs), so it keeps everything whole.
CLASS_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration"},
    "tsx": {"class_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "go": set(),
    "rust": {"impl_item"},
}


def _is_class(node, language):
    """True if node is a splittable container (possibly wrapped by decorators/export)."""
    if node.type in CLASS_TYPES.get(language, set()):
        return True
    if node.type in ("decorated_definition", "export_statement"):
        return any(_is_class(c, language) for c in node.children)
    return False


def _is_descendable(node, language):
    """Unused fallback hook; containers are handled by _split_class."""
    return False


@dataclass
class Chunk:
    text: str
    rel_path: str
    language: str
    start_line: int
    end_line: int
    symbol: str | None


_DEF_TYPES = ("function_definition", "class_definition",
              "function_declaration", "class_declaration")


def _definition_node(node):
    """Unwrap a decorated_definition / export_statement to the def it wraps."""
    if node.type in ("decorated_definition", "export_statement"):
        for c in node.children:
            if c.type in _DEF_TYPES:
                return c
    return node


def _node_name(node, source):
    """Return the name a function/class/impl node defines, if available."""
    node = _definition_node(node)
    name_node = node.child_by_field_name("name")
    if name_node is None and node.type == "impl_item":
        name_node = node.child_by_field_name("type")   # Rust: `impl Foo`
    if name_node is not None:
        return source[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace")
    return None


def _make_chunk(nodes, source, rel_path, language):
    """Build a Chunk spanning one or more adjacent AST nodes."""
    first, last = nodes[0], nodes[-1]
    text = source[first.start_byte:last.end_byte].decode("utf-8", "replace")
    symbol = _node_name(first, source) if len(nodes) == 1 else None
    return Chunk(
        text=text,
        rel_path=rel_path,
        language=language,
        start_line=first.start_point[0] + 1,
        end_line=last.end_point[0] + 1,
        symbol=symbol,
    )


def _split_class(node, source, rel_path, language, max_chars):
    """Split an oversized container into member chunks.

    Each member chunk is prefixed with the container signature and gets a
    "Container.member" symbol, so a retrieved method always says where it lives.
    The line range still points at the member's real lines.
    """
    inner = _definition_node(node)
    body = inner.child_by_field_name("body")
    class_name = _node_name(node, source) or "class"

    if body is None:                               # unusual grammar shape: keep whole
        return [_make_chunk([node], source, rel_path, language)]

    # Header = decorators/export + `class X ... {` up to (not incl.) the body.
    header = source[node.start_byte:body.start_byte].decode("utf-8", "replace").rstrip()

    members = _chunk_children(body.children, source, rel_path, language, max_chars)
    for c in members:
        c.text = f"{header}\n{c.text}"             # prepend parent context
        c.symbol = f"{class_name}.{c.symbol}" if c.symbol else class_name
    return members


def _chunk_children(children, source, rel_path, language, max_chars):
    """Greedy divide-and-combine over a list of sibling AST nodes."""
    chunks = []
    buffer = []
    buffer_len = 0

    def flush():
        nonlocal buffer, buffer_len
        if buffer:
            chunks.append(_make_chunk(buffer, source, rel_path, language))
            buffer = []
            buffer_len = 0

    for child in children:
        size = child.end_byte - child.start_byte

        if size > max_chars and _is_class(child, language):
            flush()
            chunks.extend(_split_class(child, source, rel_path, language, max_chars))
        elif size > max_chars:
            flush()
            chunks.append(_make_chunk([child], source, rel_path, language))
        elif buffer_len + size > max_chars:
            flush()
            buffer = [child]
            buffer_len = size
        else:
            buffer.append(child)
            buffer_len += size

    flush()
    return chunks


def chunk_source(source, rel_path, language, max_chars=DEFAULT_MAX_CHARS):
    """Parse source (bytes) and return a list of Chunks; [] if unsupported."""
    parser = _PARSERS.get(language)
    if parser is None:
        return []
    tree = parser.parse(source)
    chunks = _chunk_children(
        tree.root_node.children, source, rel_path, language, max_chars
    )
    return [c for c in chunks if c.text.strip()]
