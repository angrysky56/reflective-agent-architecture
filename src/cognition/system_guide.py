import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import Driver

logger = logging.getLogger(__name__)

class SystemGuideNodes:
    """
    Implements the 'Code Casefile' pattern for structural self-awareness.
    Maps the codebase into Neo4j using ConceptNodes (Casefiles) and CodeBookmarks.
    """

    def __init__(self, driver: Driver, root_path: str):
        self.driver = driver
        self.root_path = Path(root_path).resolve()
        self._ensure_schema()

    def _ensure_schema(self):
        """Create necessary constraints and indexes."""
        try:
            with self.driver.session() as session:
                # Constraints for uniqueness
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:ConceptNode) REQUIRE c.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:CodeBookmark) REQUIRE b.id IS UNIQUE")
                # Index for file path
                session.run("CREATE INDEX IF NOT EXISTS FOR (b:CodeBookmark) ON (b.file)")
        except Exception as e:
            logger.error(f"Failed to ensure schema for SystemGuideNodes: {e}")

    def create_concept(self, name: str, description: str) -> str:
        """
        Create or update a high-level ConceptNode (Casefile).
        Example: 'AuthenticationSystem', 'MemoryConsolidation'
        """
        with self.driver.session() as session:
            query = """
            MERGE (c:ConceptNode {name: $name})
            SET c.description = $description,
                c.updated_at = timestamp()
            RETURN c.name
            """
            session.run(query, name=name, description=description)
            return f"Concept '{name}' created/updated."

    def create_bookmark(self, file_path: str, line: int, snippet: str, notes: str = "") -> str:
        """
        Create a CodeBookmark pointing to specific code.
        id format: 'relative/path/to/file.py:line_number'
        """
        # Normalize path relative to root if possible
        try:
            abs_path = Path(file_path).resolve()
            if self.root_path in abs_path.parents or abs_path == self.root_path:
                rel_path = abs_path.relative_to(self.root_path)
                path_str = str(rel_path)
            else:
                # If outside root (shouldn't happen often), use absolute
                path_str = str(abs_path)
        except ValueError:
            path_str = str(file_path)

        bookmark_id = f"{path_str}:{line}"

        with self.driver.session() as session:
            query = """
            MERGE (b:CodeBookmark {id: $id})
            SET b.file = $file,
                b.line = $line,
                b.snippet = $snippet,
                b.notes = $notes,
                b.updated_at = timestamp()
            RETURN b.id
            """
            session.run(query, id=bookmark_id, file=path_str, line=line, snippet=snippet, notes=notes)
            return bookmark_id

    def link_bookmark_to_concept(self, concept_name: str, bookmark_id: str):
        """Link a bookmark to a concept (Casefile)."""
        with self.driver.session() as session:
            query = """
            MATCH (c:ConceptNode {name: $concept_name})
            MATCH (b:CodeBookmark {id: $bookmark_id})
            MERGE (c)-[:CONTAINS]->(b)
            """
            session.run(query, concept_name=concept_name, bookmark_id=bookmark_id)

    def get_concept_details(self, concept_name: str) -> Dict[str, Any]:
        """Retrieve a concept and its bookmarks."""
        with self.driver.session() as session:
            query = """
            MATCH (c:ConceptNode {name: $name})
            OPTIONAL MATCH (c)-[:CONTAINS]->(b:CodeBookmark)
            RETURN c.description as description, collect(b) as bookmarks
            """
            result = session.run(query, name=concept_name).single()
            if not result:
                return {"error": f"Concept '{concept_name}' not found."}

            bookmarks = []
            for b in result["bookmarks"]:
                if b:
                    bookmarks.append({
                        "id": b["id"],
                        "file": b["file"],
                        "line": b["line"],
                        "snippet": b["snippet"],
                        "notes": b["notes"]
                    })

            return {
                "name": concept_name,
                "description": result["description"],
                "bookmarks": bookmarks
            }

    def scan_codebase(self, sub_path: str = ".") -> str:
        """
        Auto-generate bookmarks for classes and functions in the given path.
        Creates a 'CodebaseIndex' concept and links everything to it.
        """
        target_path = self.root_path / sub_path
        if not target_path.exists():
            return f"Path {target_path} does not exist."

        self.create_concept("CodebaseIndex", "Index of all classes and functions found via scan.")

        count = 0
        # Recursive glob for python files
        files = target_path.rglob("*.py") if target_path.is_dir() else [target_path]

        for py_file in files:
            # Skip common junk
            if any(part.startswith(".") or part in ["venv", "node_modules", "build", "dist"] for part in py_file.parts):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                        # Create bookmark for definition
                        snippet = f"{node.name}"
                        notes = ast.get_docstring(node) or "No docstring"
                        # Truncate notes if too long
                        if len(notes) > 200:
                            notes = notes[:197] + "..."

                        bid = self.create_bookmark(str(py_file), node.lineno, snippet, notes)
                        self.link_bookmark_to_concept("CodebaseIndex", bid)
                        count += 1
            except Exception as e:
                logger.warning(f"Failed to parse {py_file}: {e}")

        return f"Scanned {sub_path}. Created/Updated {count} bookmarks in 'CodebaseIndex'."
