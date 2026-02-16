"""
Library Mapping Service for CodeMorph.

Manages library mappings with LLM suggestions and user verification.
Detects imports, categorizes them, and suggests target language equivalents.
"""

import json
import sys
from pathlib import Path

import yaml

from codemorph.config.models import (
    CodeMorphConfig,
    ImportInfo,
    LanguageType,
    LibraryAnalysisResult,
    LibraryMapping,
)


# Python standard library modules (3.10+)
PYTHON_STDLIB = {
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio", "asyncore",
    "atexit", "audioop", "base64", "bdb", "binascii", "binhex", "bisect",
    "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd",
    "code", "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy", "copyreg",
    "cProfile", "crypt", "csv", "ctypes", "curses", "dataclasses", "datetime",
    "dbm", "decimal", "difflib", "dis", "distutils", "doctest", "email",
    "encodings", "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt", "getpass",
    "gettext", "glob", "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "keyword", "lib2to3", "linecache",
    "locale", "logging", "lzma", "mailbox", "mailcap", "marshal", "math",
    "mimetypes", "mmap", "modulefinder", "multiprocessing", "netrc", "nis",
    "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform", "plistlib",
    "poplib", "posix", "posixpath", "pprint", "profile", "pstats", "pty", "pwd",
    "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random", "re",
    "readline", "reprlib", "resource", "rlcompleter", "runpy", "sched", "secrets",
    "select", "selectors", "shelve", "shlex", "shutil", "signal", "site",
    "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd", "sqlite3",
    "ssl", "stat", "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny", "tarfile",
    "telnetlib", "tempfile", "termios", "test", "textwrap", "threading", "time",
    "timeit", "tkinter", "token", "tokenize", "trace", "traceback", "tracemalloc",
    "tty", "turtle", "turtledemo", "types", "typing", "unicodedata", "unittest",
    "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp",
    "zipfile", "zipimport", "zlib", "zoneinfo",
}

# Java standard library package prefixes
JAVA_STDLIB_PREFIXES = {
    "java.",
    "javax.",
    "jdk.",
    "sun.",
    "com.sun.",
}


class LibraryMappingService:
    """
    Manages library mappings with LLM suggestions and user verification.

    Workflow:
    1. Load built-in mappings from YAML
    2. Analyze detected imports
    3. For unknown libraries, query LLM for suggestions
    4. Present to user for verification (warn if skipped, don't block)
    5. Use approved mappings during translation

    Usage:
        service = LibraryMappingService(config, llm_client)

        # Analyze imports from Phase 1
        result = service.analyze_imports(detected_imports)

        # Get LLM suggestions for unknown libraries
        for lib in result.unknown_libraries:
            suggestion = service.suggest_mapping(lib)

        # User verification happens in CLI
        pending = service.get_pending_for_user_review()

        # After user approves
        service.approve_mapping("requests")
    """

    def __init__(
        self,
        config: CodeMorphConfig,
        llm_client=None,  # Type hint omitted to avoid circular import
    ):
        """
        Initialize the library mapping service.

        Args:
            config: CodeMorph configuration
            llm_client: Optional LLM client for generating suggestions
        """
        self.config = config
        self.llm_client = llm_client

        # Known mappings (verified or built-in)
        self.known_mappings: dict[str, LibraryMapping] = {}

        # Pending mappings (LLM suggested, awaiting user verification)
        self.pending_mappings: dict[str, LibraryMapping] = {}

        # Skipped mappings (user chose to skip)
        self.skipped_libraries: set[str] = set()

        # Load built-in mappings
        self._load_builtin_mappings()

    def _load_builtin_mappings(self):
        """Load predefined mappings from YAML files."""
        # Determine mapping file based on translation direction
        source_lang = self.config.project.source.language
        target_lang = self.config.project.target.language

        if source_lang == LanguageType.PYTHON and target_lang == LanguageType.JAVA:
            mapping_file = "python_to_java.yaml"
        elif source_lang == LanguageType.JAVA and target_lang == LanguageType.PYTHON:
            mapping_file = "java_to_python.yaml"
        else:
            return  # Same language, no library mapping needed

        # Find the mapping file
        knowledge_dir = Path(__file__).parent
        mapping_path = knowledge_dir / "library_maps" / mapping_file

        if not mapping_path.exists():
            return

        with open(mapping_path) as f:
            data = yaml.safe_load(f)

        for mapping_data in data.get("mappings", []):
            # Support both key formats (source/target and source_library/target_library)
            source_lib = mapping_data.get("source_library") or mapping_data.get("source")
            target_lib = mapping_data.get("target_library") or mapping_data.get("target")

            # Skip invalid mappings
            if not source_lib or not target_lib:
                continue

            # Support both import key formats
            imports = (
                mapping_data.get("target_imports") or
                mapping_data.get("imports") or
                []
            )

            # Support both dependency key formats
            dependency = (
                mapping_data.get("maven_dependency") or
                mapping_data.get("maven") or
                mapping_data.get("pip_package")  # For Java to Python
            )

            mapping = LibraryMapping(
                source_library=source_lib,
                target_library=target_lib,
                target_imports=imports,
                maven_dependency=dependency,
                notes=mapping_data.get("notes"),
                verified_by_user=True,  # Built-in mappings are pre-verified
                suggested_by_llm=False,
            )
            self.known_mappings[mapping.source_library] = mapping

    def analyze_imports(
        self,
        imports: list[ImportInfo],
        project_modules: set[str] | None = None,
    ) -> LibraryAnalysisResult:
        """
        Categorize imports into known, unknown, internal, and standard library.

        Args:
            imports: List of ImportInfo from Phase 1 analysis
            project_modules: Set of module names internal to the project

        Returns:
            LibraryAnalysisResult with categorized imports
        """
        project_modules = project_modules or set()

        known: list[LibraryMapping] = []
        unknown_imports: list[ImportInfo] = []
        internal: list[str] = []
        stdlib: list[str] = []

        seen_libraries: set[str] = set()
        source_lang = self.config.project.source.language

        for imp in imports:
            # Get the module name for lookup
            module_name = imp.module if imp.module else ""
            if not module_name:
                continue

            # For deduplication, use the top-level or full module depending on language
            if source_lang == LanguageType.PYTHON:
                lookup_key = module_name.split(".")[0]
            else:
                # For Java, use full package name for known mappings check
                lookup_key = module_name

            if lookup_key in seen_libraries:
                continue
            seen_libraries.add(lookup_key)

            # Check if internal
            if imp.is_internal or lookup_key in project_modules:
                internal.append(lookup_key)
                continue

            # Check if standard library (language-specific)
            is_stdlib = self._is_stdlib_import(imp, source_lang)
            if is_stdlib:
                stdlib.append(lookup_key)
                # Some stdlib modules have mappings too (json -> Jackson, etc.)
                if lookup_key in self.known_mappings:
                    known.append(self.known_mappings[lookup_key])
                continue

            # Check if we have a known mapping
            if lookup_key in self.known_mappings:
                known.append(self.known_mappings[lookup_key])
                continue

            # Unknown library - needs LLM suggestion
            unknown_imports.append(imp)

        return LibraryAnalysisResult(
            known_mappings=known,
            unknown_imports=unknown_imports,
            internal_imports=internal,
            standard_library=stdlib,
        )

    def _is_stdlib_import(self, imp: ImportInfo, source_lang: LanguageType) -> bool:
        """Check if an import is from the standard library."""
        if imp.is_standard_library:
            return True

        module_name = imp.module

        if source_lang == LanguageType.PYTHON:
            top_level = module_name.split(".")[0] if module_name else ""
            return top_level in PYTHON_STDLIB
        elif source_lang == LanguageType.JAVA:
            # Java stdlib uses package prefixes
            for prefix in JAVA_STDLIB_PREFIXES:
                if module_name.startswith(prefix):
                    return True
            return False

        return False

    def suggest_mapping(self, source_library: str) -> LibraryMapping | None:
        """
        Query LLM to suggest a library mapping for an unknown library.

        Args:
            source_library: Name of the source library (e.g., "requests")

        Returns:
            LibraryMapping with LLM suggestion, or None if LLM unavailable
        """
        if not self.llm_client:
            return None

        target_lang = self.config.project.target.language.value
        target_version = self.config.project.target.version

        prompt = f"""I am translating Python code to {target_lang} {target_version}.

The Python code uses the library: {source_library}

What is the standard {target_lang} equivalent? Provide your answer as JSON with these fields:
- target_library: The main library/package name in {target_lang}
- maven_dependency: Maven dependency string (groupId:artifactId:version) or null if built-in
- imports: List of typical import statements needed
- notes: Brief note about any important usage differences

Example response format:
{{
    "target_library": "java.net.http.HttpClient",
    "maven_dependency": null,
    "imports": ["java.net.http.HttpClient", "java.net.http.HttpRequest", "java.net.http.HttpResponse"],
    "notes": "Built into Java 11+. Use HttpClient.newHttpClient() to create instance."
}}

Respond with ONLY the JSON, no other text."""

        try:
            response = self.llm_client.generate(prompt)

            # Parse JSON from response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()

            data = json.loads(response)

            mapping = LibraryMapping(
                source_library=source_library,
                target_library=data.get("target_library") or "",
                target_imports=data.get("imports") or [],
                maven_dependency=data.get("maven_dependency"),
                notes=data.get("notes"),
                verified_by_user=False,
                suggested_by_llm=True,
            )

            # Add to pending for user review
            self.pending_mappings[source_library] = mapping
            return mapping

        except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
            # If LLM response is malformed, create a placeholder
            mapping = LibraryMapping(
                source_library=source_library,
                target_library=f"// TODO: Find {target_lang} equivalent for {source_library}",
                target_imports=[],
                maven_dependency=None,
                notes=f"LLM suggestion failed: {e}",
                verified_by_user=False,
                suggested_by_llm=True,
            )
            self.pending_mappings[source_library] = mapping
            return mapping

    def get_pending_for_user_review(self) -> list[LibraryMapping]:
        """
        Get all mappings that need user verification.

        Returns:
            List of pending LibraryMappings
        """
        return list(self.pending_mappings.values())

    def approve_mapping(self, source_library: str):
        """
        Mark a mapping as user-verified and move to known mappings.

        Args:
            source_library: The source library name
        """
        if source_library in self.pending_mappings:
            mapping = self.pending_mappings.pop(source_library)
            mapping.verified_by_user = True
            self.known_mappings[source_library] = mapping

    def reject_mapping(self, source_library: str, user_override: LibraryMapping):
        """
        Reject LLM suggestion and use user-provided mapping.

        Args:
            source_library: The source library name
            user_override: User's corrected mapping
        """
        if source_library in self.pending_mappings:
            self.pending_mappings.pop(source_library)

        user_override.verified_by_user = True
        user_override.suggested_by_llm = False
        self.known_mappings[source_library] = user_override

    def skip_mapping(self, source_library: str):
        """
        Skip a library mapping (user chose not to provide one).

        This will show a warning but NOT block translation.

        Args:
            source_library: The source library name
        """
        if source_library in self.pending_mappings:
            self.pending_mappings.pop(source_library)
        self.skipped_libraries.add(source_library)

    def get_mapping(self, source_library: str) -> LibraryMapping | None:
        """
        Get the mapping for a library (known or pending).

        Args:
            source_library: The source library name

        Returns:
            LibraryMapping or None if not found
        """
        if source_library in self.known_mappings:
            return self.known_mappings[source_library]
        if source_library in self.pending_mappings:
            return self.pending_mappings[source_library]
        return None

    def get_all_target_imports(self) -> list[str]:
        """
        Get all target imports from known mappings.

        Useful for generating import statements in translated code.

        Returns:
            List of target import statements
        """
        imports: list[str] = []
        for mapping in self.known_mappings.values():
            imports.extend(mapping.target_imports)
        return imports

    def get_maven_dependencies(self) -> list[str]:
        """
        Get all Maven dependencies from known mappings.

        Useful for generating pom.xml.

        Returns:
            List of Maven dependency strings
        """
        deps: list[str] = []
        for mapping in self.known_mappings.values():
            if mapping.maven_dependency:
                deps.append(mapping.maven_dependency)
        return deps

    def is_stdlib(self, module_name: str) -> bool:
        """Check if a module is in Python standard library."""
        return module_name in PYTHON_STDLIB

    def save(self, output_dir: Path):
        """
        Save current mappings to a project-specific file.

        Args:
            output_dir: Directory to save mappings to
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "library_mappings.json"

        data = {
            "known_mappings": {
                k: v.model_dump() for k, v in self.known_mappings.items()
            },
            "skipped_libraries": list(self.skipped_libraries),
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, config: CodeMorphConfig, state_dir: Path, llm_client=None) -> "LibraryMappingService":
        """
        Load mappings from a project-specific file.

        Args:
            config: CodeMorph configuration
            state_dir: Directory with saved mappings
            llm_client: Optional LLM client

        Returns:
            LibraryMappingService instance
        """
        service = cls(config, llm_client)

        mapping_file = state_dir / "library_mappings.json"
        if not mapping_file.exists():
            return service

        with open(mapping_file) as f:
            data = json.load(f)

        # Load project-specific mappings (override built-ins)
        for source, mapping_data in data.get("known_mappings", {}).items():
            service.known_mappings[source] = LibraryMapping(**mapping_data)

        service.skipped_libraries = set(data.get("skipped_libraries", []))

        return service

    def get_translation_context(self) -> str:
        """
        Generate context about library mappings for LLM prompts.

        Returns:
            String with library mapping guidance for LLM
        """
        if not self.known_mappings:
            return ""

        lines = ["LIBRARY MAPPINGS (use these Java equivalents):"]
        for source, mapping in self.known_mappings.items():
            lines.append(f"  - {source} → {mapping.target_library}")
            if mapping.notes:
                lines.append(f"    Note: {mapping.notes}")

        return "\n".join(lines)
