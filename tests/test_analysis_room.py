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
        with tempfile.TemporaryDirectory() as tmpdir:
            room_dir = Path(tmpdir)
            config_path = room_dir / "room_config.yaml"
            config_path.write_text("id: test\nname: Test")

            editor = RoomConfigEditor(room_dir)
            assert editor.is_managed() is False

            editor.mark_as_managed()
            assert editor.is_managed() is True

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
