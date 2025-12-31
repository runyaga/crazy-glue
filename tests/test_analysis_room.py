"""
Tests for the Analysis Room (System Architect).

Tests the refactored command handler architecture.
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from crazy_glue.analysis import AnalysisContext
from crazy_glue.analysis import parse_command
from crazy_glue.analysis.room_editor import RoomConfigEditor
from crazy_glue.analysis.tools.graph_ops import REFERENCE_IMPLEMENTATIONS
from crazy_glue.analysis.validators import sanitize_identifier
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
def mock_installation_config():
    """Create a mock installation config."""
    config = MagicMock()
    config.room_configs = {}
    config.secrets = []
    config.get_environment = MagicMock(return_value="http://localhost:11434")
    return config


@pytest.fixture
def analysis_context(mock_installation_config, mock_agent_config):
    """Create an AnalysisContext for testing."""
    return AnalysisContext(
        installation_config=mock_installation_config,
        agent_config=mock_agent_config,
    )


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


class TestCommandParser:
    """Test the command parser."""

    def test_parse_rooms(self):
        """Test parsing 'rooms' command."""
        cmd = parse_command("rooms")
        assert cmd.command == "list_rooms"
        assert cmd.args == {}

    def test_parse_managed(self):
        """Test parsing 'managed' command."""
        cmd = parse_command("managed")
        assert cmd.command == "list_managed"

    def test_parse_inspect(self):
        """Test parsing 'inspect' command."""
        cmd = parse_command("inspect brainstorm")
        assert cmd.command == "inspect_room"
        assert cmd.args["room_id"] == "brainstorm"

    def test_parse_create_room(self):
        """Test parsing 'create room' command."""
        cmd = parse_command("create calculator room for math operations")
        assert cmd.command == "create_room"
        assert cmd.args["name"] == "calculator"
        assert cmd.args["description"] == "math operations"

    def test_parse_edit_room(self):
        """Test parsing 'edit' command."""
        cmd = parse_command("edit my-room description New description here")
        assert cmd.command == "edit_room"
        assert cmd.args["room_id"] == "my-room"
        assert cmd.args["field"] == "description"
        assert cmd.args["value"] == "New description here"

    def test_parse_generate_tool(self):
        """Test parsing 'generate tool' command."""
        cmd = parse_command("generate tool my-room my_tool List files in dir")
        assert cmd.command == "generate_tool"
        assert cmd.args["room_id"] == "my-room"
        assert cmd.args["name"] == "my_tool"
        assert cmd.args["description"] == "List files in dir"

    def test_parse_apply_tool(self):
        """Test parsing 'apply tool' command."""
        cmd = parse_command("apply tool")
        assert cmd.command == "apply_tool"

    def test_parse_discard_tool(self):
        """Test parsing 'discard tool' command."""
        cmd = parse_command("discard tool")
        assert cmd.command == "discard_tool"

    def test_parse_list_tools(self):
        """Test parsing 'list tools' command."""
        cmd = parse_command("list tools")
        assert cmd.command == "list_tools"

    def test_parse_find_entity(self):
        """Test parsing 'find' command."""
        cmd = parse_command("find BrainstormAgent")
        assert cmd.command == "find_entity"
        assert cmd.args["query"] == "brainstormagent"

    def test_parse_show_reference(self):
        """Test parsing 'show' reference command."""
        cmd = parse_command("show joker")
        assert cmd.command == "show_reference"
        assert cmd.args["name"] == "joker"

    def test_parse_add_suggestion(self):
        """Test parsing 'add suggestion' command."""
        cmd = parse_command("add suggestion my-room Try this feature")
        assert cmd.command == "add_suggestion"
        assert cmd.args["room_id"] == "my-room"
        assert cmd.args["text"] == "Try this feature"

    def test_parse_remove_suggestion(self):
        """Test parsing 'remove suggestion' command."""
        cmd = parse_command("remove suggestion my-room 2")
        assert cmd.command == "remove_suggestion"
        assert cmd.args["room_id"] == "my-room"
        assert cmd.args["index"] == "2"

    def test_parse_list_secrets(self):
        """Test parsing 'list secrets' command."""
        cmd = parse_command("list secrets")
        assert cmd.command == "list_secrets"

    def test_parse_check_secret(self):
        """Test parsing 'check secret' command."""
        cmd = parse_command("check secret API_KEY")
        assert cmd.command == "check_secret"
        assert cmd.args["name"] == "api_key"

    def test_parse_list_prompts(self):
        """Test parsing 'list prompts' command."""
        cmd = parse_command("list prompts")
        assert cmd.command == "list_prompts"

    def test_parse_unknown(self):
        """Test parsing unknown command."""
        cmd = parse_command("some random text")
        assert cmd.command == "unknown"
        assert cmd.args["input"] == "some random text"


class TestReferenceImplementations:
    """Test reference implementation lookup."""

    def test_reference_implementations_defined(self):
        """Test all expected references are defined."""
        assert "joker" in REFERENCE_IMPLEMENTATIONS
        assert "faux" in REFERENCE_IMPLEMENTATIONS
        assert "brainstorm" in REFERENCE_IMPLEMENTATIONS


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

        unique_id = f"test-{uuid.uuid4().hex[:8]}"

        with tempfile.TemporaryDirectory() as tmpdir:
            room_dir = Path(tmpdir)
            config_path = room_dir / "room_config.yaml"
            config_path.write_text(f"id: {unique_id}\nname: Test")

            editor = RoomConfigEditor(room_dir)
            assert editor.is_managed() is False

            editor.mark_as_managed()
            assert editor.is_managed() is True

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

            new_editor = RoomConfigEditor(room_dir)
            assert new_editor.get_description() == "Updated description"


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


class TestToolGeneration:
    """Test tool generation and module path handling."""

    def test_module_path_strips_src_prefix(self):
        """Test that src/ prefix is stripped from module path."""
        file_path_str = "src/crazy_glue/tools/my_tool.py"
        module_path = file_path_str
        if module_path.startswith("src/"):
            module_path = module_path[4:]
        tool_module = module_path.replace("/", ".").replace(".py", "")

        assert tool_module == "crazy_glue.tools.my_tool"
        assert not tool_module.startswith("src.")

    def test_module_path_without_src_unchanged(self):
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

    def test_tool_dotted_path_format(self):
        """Test tool_name is module.function format."""
        file_path_str = "src/crazy_glue/tools/my_tool.py"
        func_name = "my_tool"

        module_path = file_path_str
        if module_path.startswith("src/"):
            module_path = module_path[4:]
        tool_module = module_path.replace("/", ".").replace(".py", "")
        tool_dotted_path = f"{tool_module}.{func_name}"

        assert tool_dotted_path == "crazy_glue.tools.my_tool.my_tool"


class TestSanitizeIdentifier:
    """Test the sanitize_identifier function."""

    def test_hyphen_to_underscore(self):
        """Test hyphens are converted to underscores."""
        result, error = sanitize_identifier("big-filename")
        assert error is None
        assert result == "big_filename"

    def test_spaces_to_underscore(self):
        """Test spaces are converted to underscores."""
        result, error = sanitize_identifier("my tool name")
        assert error is None
        assert result == "my_tool_name"

    def test_dots_to_underscore(self):
        """Test dots are converted to underscores."""
        result, error = sanitize_identifier("file.parser")
        assert error is None
        assert result == "file_parser"

    def test_leading_digit_prefixed(self):
        """Test leading digits get underscore prefix."""
        result, error = sanitize_identifier("123tool")
        assert error is None
        assert result == "_123tool"

    def test_special_chars_removed(self):
        """Test special characters are removed."""
        result, error = sanitize_identifier("my@tool#name!")
        assert error is None
        assert result == "mytoolname"

    def test_python_keyword_suffixed(self):
        """Test Python keywords get _tool suffix."""
        result, error = sanitize_identifier("class")
        assert error is None
        assert result == "class_tool"

    def test_empty_string_error(self):
        """Test empty string returns error."""
        result, error = sanitize_identifier("")
        assert error is not None
        assert "empty" in error.lower()

    def test_only_special_chars_error(self):
        """Test string with only special chars returns error."""
        result, error = sanitize_identifier("@#$%")
        assert error is not None

    def test_uppercase_lowercased(self):
        """Test uppercase is converted to lowercase."""
        result, error = sanitize_identifier("MyToolName")
        assert error is None
        assert result == "mytoolname"

    def test_mixed_separators(self):
        """Test mixed separators are handled."""
        result, error = sanitize_identifier("my-tool.name here")
        assert error is None
        assert result == "my_tool_name_here"

    def test_multiple_underscores_collapsed(self):
        """Test multiple underscores are collapsed."""
        result, error = sanitize_identifier("my--tool__name")
        assert error is None
        assert result == "my_tool_name"


class TestAnalysisContext:
    """Test the AnalysisContext class."""

    def test_context_paths(self, analysis_context):
        """Test context derives correct paths."""
        assert analysis_context.pending_tool_path.name == "pending_tool.json"
        assert analysis_context.prompts_path.name == "prompts.json"

    def test_context_properties(self, analysis_context):
        """Test context properties from config."""
        assert analysis_context.roots == ["src/crazy_glue"]
        assert analysis_context.max_depth == 3
        assert analysis_context.max_files == 20

    def test_pending_tool_storage(self, analysis_context):
        """Test pending tool save/load/clear."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis_context.pending_tool_path = Path(tmpdir) / "pending.json"

            assert analysis_context.load_pending_tool() is None

            test_data = {"name": "test_tool", "code": "# test"}
            analysis_context.save_pending_tool(test_data)

            loaded = analysis_context.load_pending_tool()
            assert loaded["name"] == "test_tool"

            analysis_context.clear_pending_tool()
            assert analysis_context.load_pending_tool() is None

    def test_prompts_storage(self, analysis_context):
        """Test prompts save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analysis_context.prompts_path = Path(tmpdir) / "prompts.json"

            assert analysis_context.load_prompts() == {}

            prompts = {"test": {"name": "Test", "content": "Test content"}}
            analysis_context.save_prompts(prompts)

            loaded = analysis_context.load_prompts()
            assert "test" in loaded
            assert loaded["test"]["name"] == "Test"
