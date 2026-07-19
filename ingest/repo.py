"""Repo ingestion: clone a public GitHub repo and walk its source files.

This is step 1 of the pipeline. Given a GitHub URL we:
  1. shallow-clone it to a temp folder (fast — no history), and
  2. walk every file, throwing away the junk (dependencies, build output,
     binaries, huge generated files) and keeping only source code we can parse.

Everything we keep is returned as a SourceFile with the info the next step
(chunking) needs.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass

# Folders we never want to index: version control, installed dependencies,
# build/output dirs, editor/tooling caches. os.walk is told to skip these.
SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "venv", ".venv", "env", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".next", ".nuxt", "target", "vendor",
    "bin", "obj", "coverage", ".gradle", "Pods",
    ".idea", ".vscode",
}

# File extension -> language key. The key tells the chunker which tree-sitter
# grammar to use. Add a line here to support a new language.
EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
}

# Anything bigger than this is almost certainly generated or minified, not code
# a human wrote — skip it so it doesn't pollute retrieval.
MAX_FILE_BYTES = 1_000_000  # 1 MB


@dataclass
class SourceFile:
    path: str       # absolute path on disk
    rel_path: str   # path relative to the repo root — this is what we cite
    language: str   # key into EXT_TO_LANG / the chunker's grammars


def clone_repo(url, dest=None):
    """Shallow-clone `url` into a temp folder (or `dest`). Returns the path.

    --depth 1 grabs only the latest commit, so cloning a big repo is quick.
    """
    dest = dest or tempfile.mkdtemp(prefix="rag_repo_")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, dest],
        check=True, capture_output=True, text=True,
    )
    return dest


def iter_source_files(root):
    """Yield a SourceFile for each supported, non-junk code file under `root`."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs IN PLACE. Editing dirnames here stops os.walk from
        # ever descending into node_modules/.git/etc — much faster than
        # visiting them and filtering later.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            language = EXT_TO_LANG.get(ext)
            if language is None:
                continue  # not a language we chunk (yet)

            full = os.path.join(dirpath, name)
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue  # file vanished / unreadable — skip

            yield SourceFile(
                path=full,
                rel_path=os.path.relpath(full, root),
                language=language,
            )
