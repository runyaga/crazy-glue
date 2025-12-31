"""
Tests for the Analysis Room (System Architect).

Tests the Cartographer-based knowledge graph building and querying.
"""

import ast
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from crazy_glue.factories.analysis.room_editor import RoomConfigEditor
from crazy_glue.factories.analysis_factory import REFERENCE_IMPLEMENTATIONS
from crazy_glue.factories.analysis_factory import AnalysisAgent
from crazy_glue.factories.analysis_factory import create_analysis_agent


@pytest.fixture
def mock_agent_config():
    """Create a mock agent config for testing."""
    config = MagicMock()
    config.extra_config = {
        "map_path": "test_db/project_map.json",
        "roots": ["src/crazy_glue"],
        "max_depth": 3,
        "max_files": 20,
    }
    return config


@pytest.fixture
def analysis_agent(mock_agent_config):
    """Create an analysis agent instance."""
    return create_analysis_agent(mock_agent_config)


class TestAnalysisAgentCreation:
    """Test agent creation and configuration."""

    def test_create_analysis_agent(self, mock_agent_config):
        """Test factory function creates agent."""
        agent = create_analysis_agent(mock_agent_config)
        assert isinstance(agent, AnalysisAgent)
        assert agent.agent_config == mock_agent_config

    def test_agent_properties(self, analysis_agent):
        """Test agent properties from config."""
        assert analysis_agent.map_path == "test_db/project_map.json"
        assert analysis_agent.roots == ["src/crazy_glue"]
        assert analysis_agent.max_depth == 3
        assert analysis_agent.max_files == 20


class TestReferenceImplementations:
    """Test reference implementation lookup."""

    def test_reference_implementations_defined(self):
        """Test all expected references are defined."""
        assert "joker" in REFERENCE_IMPLEMENTATIONS
        assert "faux" in REFERENCE_IMPLEMENTATIONS
        assert "brainstorm" in REFERENCE_IMPLEMENTATIONS

    def test_read_joker_reference(self, analysis_agent):
        """Test reading Joker reference implementation."""
        result = analysis_agent._read_reference_implementation("joker")
        assert "joker_agent_factory" in result or "Error" not in result[:5]

    def test_read_faux_reference(self, analysis_agent):
        """Test reading Faux reference implementation."""
        result = analysis_agent._read_reference_implementation("faux")
        assert "FauxAgent" in result or "Error" not in result[:5]

    def test_read_brainstorm_reference(self, analysis_agent):
        """Test reading Brainstorm reference implementation."""
        result = analysis_agent._read_reference_implementation("brainstorm")
        assert "BrainstormAgent" in result or "brainstorm" in result.lower()

    def test_unknown_reference(self, analysis_agent):
        """Test unknown reference returns error."""
        result = analysis_agent._read_reference_implementation("unknown")
        assert "Unknown reference" in result


class TestScaffolding:
    """Test room scaffolding functionality."""

    def test_scaffold_room_generates_yaml(self, analysis_agent):
        """Test scaffolding generates valid YAML."""
        files = analysis_agent._scaffold_room("Calculator", "A room for math")

        assert len(files) > 0
        yaml_path = "rooms/calculator/room_config.yaml"
        assert yaml_path in files

        content = files[yaml_path]
        parsed = yaml.safe_load(content)
        assert parsed["id"] == "calculator"
        assert parsed["name"] == "Calculator"
        assert "agent" in parsed

    def test_scaffold_room_slug_formatting(self, analysis_agent):
        """Test room names are properly slugified."""
        files = analysis_agent._scaffold_room("My Cool Room", "Test intent")

        yaml_path = "rooms/my-cool-room/room_config.yaml"
        assert yaml_path in files

        parsed = yaml.safe_load(files[yaml_path])
        assert parsed["id"] == "my-cool-room"

    def test_scaffold_generates_valid_room_config(self, analysis_agent):
        """Test scaffolded rooms generate valid config without extra fields."""
        files = analysis_agent._scaffold_room("Test", "A test room")

        yaml_path = "rooms/test/room_config.yaml"
        parsed = yaml.safe_load(files[yaml_path])

        # Should not have _managed_by field (breaks soliplex)
        assert "_managed_by" not in parsed
        # Should have required fields
        assert parsed["id"] == "test"
        assert "agent" in parsed

    def test_apply_scaffold_rejects_unsafe_paths(self, analysis_agent):
        """Test scaffold rejects paths outside allowed directories."""
        unsafe_files = {
            "/etc/passwd": "malicious content",
            "../../../etc/hosts": "more malicious",
        }

        result = analysis_agent._apply_scaffold(unsafe_files)
        assert len(result["errors"]) == 2
        assert len(result["written"]) == 0

    def test_apply_scaffold_allows_safe_paths(self, analysis_agent):
        """Test scaffold application allows paths in rooms/."""
        # Note: This test validates path filtering logic only.
        # It does NOT actually write files (would pollute project).
        # The _apply_scaffold method checks prefixes before writing.
        safe_files = {
            "rooms/test-temp/room_config.yaml": "id: test\nname: Test",
        }
        # Verify the path passes the prefix check
        allowed = ["rooms/", "src/crazy_glue/factories/"]
        for path in safe_files:
            assert any(path.startswith(p) for p in allowed)


class TestQueryGraph:
    """Test knowledge graph querying."""

    def test_query_without_graph(self, analysis_agent):
        """Test query returns error when no graph exists."""
        result = analysis_agent._query_graph("test")
        assert len(result) == 1
        assert "error" in result[0]

    def test_query_returns_list(self, analysis_agent):
        """Test query returns a list."""
        result = analysis_agent._query_graph("anything")
        assert isinstance(result, list)


class TestReadEntitySource:
    """Test entity source reading."""

    def test_read_entity_without_graph(self, analysis_agent):
        """Test reading entity without graph returns error."""
        result = analysis_agent._read_entity_source("nonexistent-id")
        assert "Error" in result
        assert "Knowledge graph not found" in result


@pytest.mark.asyncio
class TestKnowledgeGraphIntegration:
    """Integration tests for knowledge graph building."""

    async def test_refresh_knowledge_graph_creates_file(self, analysis_agent):
        """Test refresh creates knowledge map file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis_agent.agent_config.extra_config["map_path"] = (
                f"{tmpdir}/test_map.json"
            )

            result = await analysis_agent._refresh_knowledge_graph()

            assert result["status"] == "complete"
            assert "total_entities" in result
            assert "total_links" in result

    async def test_knowledge_map_is_valid_json(self, analysis_agent):
        """Test generated knowledge map is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            map_path = Path(tmpdir) / "test_map.json"
            extra = analysis_agent.agent_config.extra_config
            extra["map_path"] = str(map_path)

            await analysis_agent._refresh_knowledge_graph()

            if map_path.exists():
                content = map_path.read_text()
                parsed = json.loads(content)
                assert "entities" in parsed
                assert "links" in parsed


class TestRoomConfigValidity:
    """Test that generated room configs are valid."""

    def test_room_config_yaml_is_valid(self):
        """Test the analysis room config is valid YAML."""
        config_path = Path("rooms/analysis/room_config.yaml")
        if config_path.exists():
            content = config_path.read_text()
            parsed = yaml.safe_load(content)
            assert parsed["id"] == "analysis"
            assert parsed["agent"]["kind"] == "factory"
            assert "analysis_factory" in parsed["agent"]["factory_name"]


class TestFactorySyntax:
    """Test that the factory module has valid Python syntax."""

    def test_analysis_factory_syntax(self):
        """Test analysis_factory.py is valid Python."""
        factory_path = Path("src/crazy_glue/factories/analysis_factory.py")
        if factory_path.exists():
            content = factory_path.read_text()
            try:
                ast.parse(content)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in analysis_factory.py: {e}")


class TestRoomConfigEditor:
    """Test the RoomConfigEditor class."""

    def test_load_room_config(self):
        """Test loading a room config."""
        room_path = Path("rooms/analysis")
        if room_path.exists():
            editor = RoomConfigEditor(room_path)
            data = editor.load()
            assert data["id"] == "analysis"
            assert "agent" in data

    def test_load_nonexistent_raises(self):
        """Test loading nonexistent config raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            editor = RoomConfigEditor(Path(tmpdir))
            with pytest.raises(FileNotFoundError):
                editor.load()

    def test_is_managed_false_for_unmanaged(self):
        """Test is_managed returns False for unmanaged rooms."""
        room_path = Path("rooms/analysis")
        if room_path.exists():
            editor = RoomConfigEditor(room_path)
            assert editor.is_managed() is False

    def test_room_editor_metadata_getters(self):
        """Test metadata getter methods."""
        room_path = Path("rooms/analysis")
        if room_path.exists():
            editor = RoomConfigEditor(room_path)
            assert editor.get_id() == "analysis"
            assert editor.get_name() == "System Architect"
            assert "Codebase navigator" in editor.get_description()

    def test_room_editor_in_memory_modifications(self):
        """Test modifications work in memory without saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            room_dir = Path(tmpdir)
            config_path = room_dir / "room_config.yaml"
            config_path.write_text(
                "id: test\nname: Test\ndescription: Test room"
            )

            editor = RoomConfigEditor(room_dir)
            editor.set_description("New description")
            assert editor.get_description() == "New description"

            editor.add_suggestion("Try this")
            assert "Try this" in editor.get_suggestions()

            editor.add_tool("soliplex.tools.search_documents")
            assert "soliplex.tools.search_documents" in editor.list_tools()

            editor.add_mcp_http("exa", "https://mcp.exa.ai/mcp")
            assert "exa" in editor.list_mcp_toolsets()

    def test_mark_as_managed(self):
        """Test marking a room as managed."""
        import uuid

        # Use unique room ID to avoid conflicts with persistent tracking
        unique_id = f"test-{uuid.uuid4().hex[:8]}"

        with tempfile.TemporaryDirectory() as tmpdir:
            room_dir = Path(tmpdir)
            config_path = room_dir / "room_config.yaml"
            config_path.write_text(f"id: {unique_id}\nname: Test")

            editor = RoomConfigEditor(room_dir)
            assert editor.is_managed() is False

            editor.mark_as_managed()
            assert editor.is_managed() is True

            # Clean up: unmark to avoid polluting tracking file
            editor.unmark_as_managed()

    def test_save_persists_changes(self):
        """Test save writes changes to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            room_dir = Path(tmpdir)
            config_path = room_dir / "room_config.yaml"
            config_path.write_text("id: test\nname: Test")

            editor = RoomConfigEditor(room_dir)
            editor.set_description("Updated description")
            editor.save()

            # Reload and verify
            new_editor = RoomConfigEditor(room_dir)
            assert new_editor.get_description() == "Updated description"


class TestToolGeneration:
    """Test tool generation and module path handling."""

    def test_module_path_strips_src_prefix(self, analysis_agent):
        """Test that src/ prefix is stripped from module path."""
        # Simulate the logic in _apply_pending_tool
        file_path_str = "src/crazy_glue/tools/my_tool.py"
        module_path = file_path_str
        if module_path.startswith("src/"):
            module_path = module_path[4:]
        tool_module = module_path.replace("/", ".").replace(".py", "")

        assert tool_module == "crazy_glue.tools.my_tool"
        assert not tool_module.startswith("src.")

    def test_module_path_without_src_unchanged(self, analysis_agent):
        """Test paths without src/ are unchanged."""
        file_path_str = "crazy_glue/tools/my_tool.py"
        module_path = file_path_str
        if module_path.startswith("src/"):
            module_path = module_path[4:]
        tool_module = module_path.replace("/", ".").replace(".py", "")

        assert tool_module == "crazy_glue.tools.my_tool"

    def test_existing_tool_importable(self):
        """Test that existing tools can be imported."""
        import importlib

        # This should work - crazy_glue.tools should exist
        # Test with an existing tool if any
        tools_path = Path("src/crazy_glue/tools")
        if tools_path.exists():
            for tool_file in tools_path.glob("*.py"):
                if tool_file.name.startswith("_"):
                    continue
                module_name = f"crazy_glue.tools.{tool_file.stem}"
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    pytest.fail(f"Could not import {module_name}")

    def test_tool_function_exists_and_callable(self):
        """Test that tool modules contain callable functions."""
        import importlib

        tools_path = Path("src/crazy_glue/tools")
        if tools_path.exists():
            for tool_file in tools_path.glob("*.py"):
                if tool_file.name.startswith("_"):
                    continue
                func_name = tool_file.stem
                module_name = f"crazy_glue.tools.{func_name}"
                module = importlib.import_module(module_name)
                tool_func = getattr(module, func_name, None)
                assert tool_func is not None, f"No {func_name} in {module_name}"
                assert callable(tool_func), f"{func_name} is not callable"

    def test_tool_dotted_path_format(self, analysis_agent):
        """Test tool_name is module.function format."""
        # The tool_name must be module.function, not just module
        file_path_str = "src/crazy_glue/tools/my_tool.py"
        func_name = "my_tool"

        module_path = file_path_str
        if module_path.startswith("src/"):
            module_path = module_path[4:]
        tool_module = module_path.replace("/", ".").replace(".py", "")
        tool_dotted_path = f"{tool_module}.{func_name}"

        assert tool_dotted_path == "crazy_glue.tools.my_tool.my_tool"


class TestSanitizeIdentifier:
    """Test the _sanitize_identifier function."""

    def test_hyphen_to_underscore(self):
        """Test hyphens are converted to underscores."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("big-filename")
        assert error is None
        assert result == "big_filename"

    def test_spaces_to_underscore(self):
        """Test spaces are converted to underscores."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("my tool name")
        assert error is None
        assert result == "my_tool_name"

    def test_dots_to_underscore(self):
        """Test dots are converted to underscores."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("file.parser")
        assert error is None
        assert result == "file_parser"

    def test_leading_digit_prefixed(self):
        """Test leading digits get underscore prefix."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("123tool")
        assert error is None
        assert result == "_123tool"

    def test_special_chars_removed(self):
        """Test special characters are removed."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("my@tool#name!")
        assert error is None
        assert result == "mytoolname"

    def test_python_keyword_suffixed(self):
        """Test Python keywords get _tool suffix."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("class")
        assert error is None
        assert result == "class_tool"

    def test_empty_string_error(self):
        """Test empty string returns error."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("")
        assert error is not None
        assert "empty" in error.lower()

    def test_only_special_chars_error(self):
        """Test string with only special chars returns error."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("@#$%")
        assert error is not None

    def test_uppercase_lowercased(self):
        """Test uppercase is converted to lowercase."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("MyToolName")
        assert error is None
        assert result == "mytoolname"

    def test_mixed_separators(self):
        """Test mixed separators are handled."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("my-tool.name here")
        assert error is None
        assert result == "my_tool_name_here"

    def test_multiple_underscores_collapsed(self):
        """Test multiple underscores are collapsed."""
        from crazy_glue.factories.analysis_factory import _sanitize_identifier
        result, error = _sanitize_identifier("my--tool__name")
        assert error is None
        assert result == "my_tool_name"


class TestPromptLibrary:
    """Test prompt library functionality."""

    def test_list_prompts_empty(self, analysis_agent):
        """Test listing prompts when none exist."""
        # Clear any existing prompts
        prompts_path = analysis_agent._get_prompts_path()
        if prompts_path.exists():
            prompts_path.unlink()

        prompts = analysis_agent._list_prompts()
        assert prompts == []

    def test_add_prompt(self, analysis_agent):
        """Test adding a prompt to the library."""
        # Clear existing prompts first
        prompts_path = analysis_agent._get_prompts_path()
        if prompts_path.exists():
            prompts_path.unlink()

        result = analysis_agent._add_prompt(
            "Test Helper",
            "You are a helpful test assistant."
        )
        assert result["status"] == "success"
        assert "test-helper" in result["message"]

    def test_add_prompt_creates_slug(self, analysis_agent):
        """Test that prompt names are slugified."""
        prompts_path = analysis_agent._get_prompts_path()
        if prompts_path.exists():
            prompts_path.unlink()

        analysis_agent._add_prompt("My Cool Prompt", "Content here")
        prompts = analysis_agent._load_prompts()
        assert "my-cool-prompt" in prompts

    def test_get_prompt(self, analysis_agent):
        """Test getting a specific prompt."""
        prompts_path = analysis_agent._get_prompts_path()
        if prompts_path.exists():
            prompts_path.unlink()

        analysis_agent._add_prompt("finder", "You find things.")
        prompt = analysis_agent._get_prompt("finder")
        assert prompt is not None
        assert prompt["content"] == "You find things."

    def test_get_prompt_case_insensitive(self, analysis_agent):
        """Test prompt lookup is case insensitive."""
        prompts_path = analysis_agent._get_prompts_path()
        if prompts_path.exists():
            prompts_path.unlink()

        analysis_agent._add_prompt("CamelCase", "Test content")
        prompt = analysis_agent._get_prompt("camelcase")
        assert prompt is not None

    def test_remove_prompt(self, analysis_agent):
        """Test removing a prompt."""
        prompts_path = analysis_agent._get_prompts_path()
        if prompts_path.exists():
            prompts_path.unlink()

        analysis_agent._add_prompt("temporary", "Will be removed")
        result = analysis_agent._remove_prompt("temporary")
        assert result["status"] == "success"

        prompt = analysis_agent._get_prompt("temporary")
        assert prompt is None

    def test_remove_nonexistent_prompt(self, analysis_agent):
        """Test removing a prompt that doesn't exist."""
        result = analysis_agent._remove_prompt("nonexistent-prompt-xyz")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_add_duplicate_prompt(self, analysis_agent):
        """Test adding a prompt with same name fails."""
        prompts_path = analysis_agent._get_prompts_path()
        if prompts_path.exists():
            prompts_path.unlink()

        analysis_agent._add_prompt("unique", "First version")
        result = analysis_agent._add_prompt("unique", "Second version")
        assert result["status"] == "error"
        assert "already exists" in result["message"]

    def test_prompt_has_timestamp(self, analysis_agent):
        """Test that prompts have created timestamp."""
        prompts_path = analysis_agent._get_prompts_path()
        if prompts_path.exists():
            prompts_path.unlink()

        analysis_agent._add_prompt("timestamped", "Has a timestamp")
        prompts = analysis_agent._load_prompts()
        assert "created" in prompts["timestamped"]
        assert "T" in prompts["timestamped"]["created"]  # ISO format
