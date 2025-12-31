# Milestone 4: Documentation & Tests

## Prerequisites

- [Milestone 3](./03-write-operations.md) complete (all gates passed)

## Objective

Complete test coverage and documentation for all 20 MCP tools.

## Deliverables

### 1. Complete Test Suite

**File:** `tests/test_mcp_tools.py`

```python
"""Tests for Analysis Room MCP tools."""
import pytest
from unittest.mock import MagicMock, patch

from crazy_glue.analysis import mcp_tools


# --- Fixtures ---

@pytest.fixture
def mock_tool_config():
    """Create mock tool_config with installation context."""
    config = MagicMock()
    config._installation_config.room_configs = {
        "analysis": MagicMock(agent_config=MagicMock(extra_config={})),
        "research": MagicMock(),
    }
    return config


@pytest.fixture
def mock_context():
    """Create mock AnalysisContext."""
    return MagicMock()


# --- Read Operation Tests ---

class TestReadOperations:

    def test_architect_list_rooms(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.list_rooms") as mock:
            mock.return_value = [
                {"id": "analysis", "name": "Analysis", "description": "Test", "agent_kind": "factory"},
            ]
            result = mcp_tools.architect_list_rooms(mock_tool_config)

        assert isinstance(result, mcp_tools.RoomListResult)
        assert result.count == 1
        assert result.rooms[0].id == "analysis"

    def test_architect_list_managed(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.list_managed_rooms") as mock:
            mock.return_value = []
            result = mcp_tools.architect_list_managed(mock_tool_config)

        assert isinstance(result, mcp_tools.RoomListResult)
        assert result.count == 0

    def test_architect_inspect_room_found(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.inspect_room") as mock:
            mock.return_value = {
                "id": "analysis",
                "name": "Analysis",
                "description": "Test",
                "welcome_message": "Welcome",
                "suggestions": ["test"],
                "agent": {"kind": "factory"},
                "config_path": "/path/to/config",
            }
            result = mcp_tools.architect_inspect_room("analysis", mock_tool_config)

        assert isinstance(result, mcp_tools.RoomDetailsResult)
        assert result.id == "analysis"
        assert result.error is None

    def test_architect_inspect_room_not_found(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.inspect_room") as mock:
            mock.return_value = {"error": "Room not found"}
            result = mcp_tools.architect_inspect_room("nonexistent", mock_tool_config)

        assert result.error == "Room not found"

    def test_architect_list_tools(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.list_available_tools") as mock:
            mock.return_value = [
                {"name": "search_documents", "description": "RAG search", "requires": "rag_lancedb_stem"},
            ]
            result = mcp_tools.architect_list_tools(mock_tool_config)

        assert isinstance(result, mcp_tools.ToolListResult)
        assert len(result.tools) == 1

    def test_architect_list_secrets(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.list_secrets") as mock:
            mock.return_value = [
                {"name": "OPENAI_API_KEY", "resolved": True, "sources": ["env"]},
            ]
            result = mcp_tools.architect_list_secrets(mock_tool_config)

        assert isinstance(result, mcp_tools.SecretListResult)
        assert result.secrets[0].name == "OPENAI_API_KEY"

    def test_architect_check_secret(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.check_secret") as mock:
            mock.return_value = {
                "name": "TEST_KEY",
                "configured": True,
                "resolved": True,
                "sources": ["env"],
            }
            result = mcp_tools.architect_check_secret("TEST_KEY", mock_tool_config)

        assert isinstance(result, mcp_tools.SecretCheckResult)
        assert result.resolved is True

    def test_architect_list_prompts(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.list_prompts") as mock:
            mock.return_value = [
                {"id": "1", "name": "test", "preview": "...", "created": "2024-01-01"},
            ]
            result = mcp_tools.architect_list_prompts(mock_tool_config)

        assert isinstance(result, mcp_tools.PromptListResult)
        assert len(result.prompts) == 1

    def test_architect_show_prompt_found(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.get_prompt") as mock:
            mock.return_value = {"name": "test", "content": "Test content", "created": "2024-01-01"}
            result = mcp_tools.architect_show_prompt("test", mock_tool_config)

        assert isinstance(result, mcp_tools.PromptDetailResult)
        assert result.content == "Test content"
        assert result.error is None

    def test_architect_show_prompt_not_found(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.get_prompt") as mock:
            mock.return_value = None
            result = mcp_tools.architect_show_prompt("nonexistent", mock_tool_config)

        assert result.error is not None

    def test_architect_find_entity(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.query_graph") as mock:
            mock.return_value = [
                {"id": "1", "name": "TestClass", "type": "class", "summary": "Test", "location": "test.py", "line": 10},
            ]
            result = mcp_tools.architect_find_entity("TestClass", mock_tool_config)

        assert isinstance(result, mcp_tools.SearchResult)
        assert result.count == 1

    def test_architect_show_reference(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.read_reference_implementation") as mock:
            mock.return_value = "class Joker: pass"
            result = mcp_tools.architect_show_reference("joker", mock_tool_config)

        assert isinstance(result, mcp_tools.ReferenceResult)
        assert "Joker" in result.source
        assert result.error is None


# --- Write Operation Tests ---

class TestWriteOperations:

    def test_architect_create_room(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.create_room") as mock:
            mock.return_value = {"room_id": "new-room", "config_path": "/path", "status": "success"}
            result = mcp_tools.architect_create_room("new-room", "Description", mock_tool_config)

        assert isinstance(result, mcp_tools.CreateRoomResult)
        assert result.status == "success"

    def test_architect_edit_room(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.edit_room") as mock:
            mock.return_value = {"status": "success", "message": "Updated"}
            result = mcp_tools.architect_edit_room("room", "description", "New desc", mock_tool_config)

        assert isinstance(result, mcp_tools.StatusResult)
        assert result.status == "success"

    def test_architect_add_tool(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.manage_tool") as mock:
            mock.return_value = {"status": "success", "message": "Tool added"}
            result = mcp_tools.architect_add_tool("room", "tool_name", mock_tool_config)

        assert result.status == "success"

    def test_architect_remove_tool(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.manage_tool") as mock:
            mock.return_value = {"status": "success", "message": "Tool removed"}
            result = mcp_tools.architect_remove_tool("room", "tool_name", mock_tool_config)

        assert result.status == "success"

    def test_architect_add_prompt(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.add_prompt") as mock:
            mock.return_value = {"status": "success", "message": "Prompt saved"}
            result = mcp_tools.architect_add_prompt("name", "content", mock_tool_config)

        assert result.status == "success"

    def test_architect_remove_prompt(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.remove_prompt") as mock:
            mock.return_value = {"status": "success", "message": "Prompt deleted"}
            result = mcp_tools.architect_remove_prompt("name", mock_tool_config)

        assert result.status == "success"

    def test_architect_add_mcp_http(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.manage_mcp") as mock:
            mock.return_value = {"status": "success", "message": "MCP added"}
            result = mcp_tools.architect_add_mcp_http("room", "name", "http://url", mock_tool_config)

        assert result.status == "success"

    def test_architect_add_mcp_stdio(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.manage_mcp") as mock:
            mock.return_value = {"status": "success", "message": "MCP added"}
            result = mcp_tools.architect_add_mcp_stdio("room", "name", "cmd", mock_tool_config)

        assert result.status == "success"

    def test_architect_remove_mcp(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.manage_mcp") as mock:
            mock.return_value = {"status": "success", "message": "MCP removed"}
            result = mcp_tools.architect_remove_mcp("room", "name", mock_tool_config)

        assert result.status == "success"

    def test_architect_toggle_mcp_server(self, mock_tool_config):
        with patch("crazy_glue.analysis.tools.toggle_mcp_server") as mock:
            mock.return_value = {"status": "success", "message": "MCP server enabled"}
            result = mcp_tools.architect_toggle_mcp_server("room", True, mock_tool_config)

        assert result.status == "success"
```

### 2. Update Soliplex Lessons Learned

**File:** `docs/specs/soliplex-lessons.md`

After completing all milestones, document:
- Wrapper patterns that worked
- Context building approach
- Integration gotchas
- Performance observations

### 3. Update Main Spec

**File:** `docs/specs/analysis-mcp-tools.md`

- Mark as implemented
- Add any deviations from original design
- Document actual endpoint behavior

## Gating Criteria

### Gate 1: All Unit Tests Pass
```bash
pytest tests/test_mcp_tools.py -v --tb=short
```
**Expected:** All 22 tests pass

### Gate 2: Test Coverage > 80%
```bash
pytest tests/test_mcp_tools.py --cov=crazy_glue.analysis.mcp_tools --cov-report=term-missing
```
**Expected:** Coverage >= 80%

### Gate 3: Soliplex Lessons Updated
```bash
test -f docs/specs/soliplex-lessons.md && echo "OK"
```
**Expected:** `OK`

### Gate 4: Integration Test Passes
```bash
# Full integration test
./scripts/test_mcp_integration.sh
```
**Expected:** All MCP calls succeed

## Success Criteria

| Gate | Description | Pass/Fail |
|------|-------------|-----------|
| 1 | All unit tests pass | |
| 2 | Coverage >= 80% | |
| 3 | Lessons learned documented | |
| 4 | Integration test passes | |

## Project Complete

Once all gates pass, the Analysis Room MCP Tools feature is complete.

Update `docs/specs/milestones/00-overview.md` to mark all milestones as Complete.
