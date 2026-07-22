from ingest.chunker import chunk_source


def test_python_functions_and_class_no_broken_chunks():
    src = b"class Foo:\n    def bar(self):\n        return 1\n\ndef baz():\n    return 2\n"
    chunks = chunk_source(src, "a.py", "python")
    assert chunks
    assert all(c.rel_path == "a.py" and c.start_line >= 1 for c in chunks)
    assert not any(c.text.strip().startswith("def ") and c.text.rstrip().endswith(":")
                   for c in chunks)


def test_unsupported_language_returns_empty():
    assert chunk_source(b"SELECT 1;", "x.sql", "sql") == []


def test_go_and_java_are_parsed():
    go = chunk_source(b"package main\nfunc Add(a int) int { return a }\n", "m.go", "go")
    assert go and any("Add" in c.text for c in go)
    java = chunk_source(b"class Foo { int bar() { return 1; } }\n", "F.java", "java")
    assert java and any("bar" in c.text for c in java)


def test_js_assigned_function_names():
    """CommonJS / arrow-assigned functions should still get symbol names."""
    src = (b"const slugify = (s) => s.toLowerCase();\n"
           b"module.exports = function createApp() { return 2; };\n"
           b"exports.render = (v) => v;\n"
           b"function plain() { return 4; }\n")
    # max_chars=1 forces each top-level statement into its own chunk
    chunks = chunk_source(src, "a.js", "javascript", max_chars=1)
    names = {c.symbol for c in chunks if c.symbol}
    assert {"slugify", "module.exports", "exports.render", "plain"} <= names
