"""
MCP Server Registry and Selection Logic
Dynamic discovery and selection of MCP servers based on task requirements
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import asyncio

logger = logging.getLogger(__name__)


class MCPCapability(Enum):
    """Standard MCP server capabilities"""

    # Core capabilities
    FILE_OPERATIONS = "file_operations"
    CODE_EXECUTION = "code_execution"
    WEB_SEARCH = "web_search"
    API_CALLS = "api_calls"
    DATA_ANALYSIS = "data_analysis"

    # Advanced capabilities
    BROWSER_AUTOMATION = "browser_automation"
    DATABASE_QUERIES = "database_queries"
    MACHINE_LEARNING = "machine_learning"
    IMAGE_PROCESSING = "image_processing"
    NATURAL_LANGUAGE = "natural_language"

    # Specialized capabilities
    GITHUB_INTEGRATION = "github_integration"
    DOCKER_MANAGEMENT = "docker_management"
    CLOUD_SERVICES = "cloud_services"
    MONITORING = "monitoring"
    SECURITY_SCANNING = "security_scanning"


@dataclass
class MCPServer:
    """Represents an MCP server with its capabilities"""

    name: str
    version: str
    endpoint: str  # Connection endpoint
    protocol: str = "stdio"  # stdio, http, websocket

    # Capabilities
    capabilities: Set[MCPCapability] = field(default_factory=set)

    # Performance metrics
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    max_concurrent_requests: int = 10

    # Configuration
    requires_auth: bool = False
    api_key: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # Status
    is_available: bool = True
    last_health_check: Optional[float] = None


class MCPServerRegistry:
    """Registry for discovering and managing MCP servers"""

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self._initialize_default_servers()
        self._discover_servers()

    def _initialize_default_servers(self):
        """Initialize with known MCP servers"""

        # File system server
        self.servers["filesystem"] = MCPServer(
            name="filesystem",
            version="1.0.0",
            endpoint="mcp-filesystem",
            capabilities={
                MCPCapability.FILE_OPERATIONS,
            },
            avg_latency_ms=5.0
        )

        # Code execution server
        self.servers["code-runner"] = MCPServer(
            name="code-runner",
            version="1.0.0",
            endpoint="mcp-code-runner",
            capabilities={
                MCPCapability.CODE_EXECUTION,
            },
            avg_latency_ms=100.0,
            config={
                "languages": ["python", "javascript", "typescript", "rust"],
                "timeout": 30000
            }
        )

        # Web search server
        self.servers["web-search"] = MCPServer(
            name="web-search",
            version="1.0.0",
            endpoint="mcp-web-search",
            capabilities={
                MCPCapability.WEB_SEARCH,
            },
            avg_latency_ms=500.0,
            requires_auth=True
        )

        # GitHub server
        self.servers["github"] = MCPServer(
            name="github",
            version="1.0.0",
            endpoint="mcp-github",
            capabilities={
                MCPCapability.GITHUB_INTEGRATION,
                MCPCapability.FILE_OPERATIONS,
            },
            avg_latency_ms=200.0,
            requires_auth=True
        )

        # Browser automation server
        self.servers["playwright"] = MCPServer(
            name="playwright",
            version="1.0.0",
            endpoint="mcp-playwright",
            capabilities={
                MCPCapability.BROWSER_AUTOMATION,
                MCPCapability.WEB_SEARCH,
            },
            avg_latency_ms=1000.0,
            config={
                "headless": True,
                "browser": "chromium"
            }
        )

        # Data analysis server
        self.servers["data-analyst"] = MCPServer(
            name="data-analyst",
            version="1.0.0",
            endpoint="mcp-data-analyst",
            capabilities={
                MCPCapability.DATA_ANALYSIS,
                MCPCapability.MACHINE_LEARNING,
            },
            avg_latency_ms=300.0,
            config={
                "frameworks": ["pandas", "numpy", "scikit-learn", "tensorflow"]
            }
        )

        # Database server
        self.servers["database"] = MCPServer(
            name="database",
            version="1.0.0",
            endpoint="mcp-database",
            capabilities={
                MCPCapability.DATABASE_QUERIES,
            },
            avg_latency_ms=50.0,
            config={
                "databases": ["postgresql", "mysql", "sqlite", "mongodb"]
            }
        )

        # Docker management server
        self.servers["docker"] = MCPServer(
            name="docker",
            version="1.0.0",
            endpoint="mcp-docker",
            capabilities={
                MCPCapability.DOCKER_MANAGEMENT,
                MCPCapability.CODE_EXECUTION,
            },
            avg_latency_ms=2000.0
        )

        # Security scanning server
        self.servers["security"] = MCPServer(
            name="security",
            version="1.0.0",
            endpoint="mcp-security",
            capabilities={
                MCPCapability.SECURITY_SCANNING,
            },
            avg_latency_ms=1500.0,
            config={
                "scanners": ["semgrep", "bandit", "safety", "trivy"]
            }
        )

    def _discover_servers(self):
        """Discover additional MCP servers from configuration"""
        try:
            # Check for MCP server configuration file
            config_path = Path.home() / ".config" / "mcp" / "servers.json"

            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                for server_id, server_config in config.items():
                    if server_id not in self.servers:
                        self.servers[server_id] = MCPServer(
                            name=server_config.get("name", server_id),
                            version=server_config.get("version", "unknown"),
                            endpoint=server_config.get("endpoint", server_id),
                            protocol=server_config.get("protocol", "stdio"),
                            capabilities={
                                MCPCapability[cap.upper()]
                                for cap in server_config.get("capabilities", [])
                                if cap.upper() in MCPCapability.__members__
                            },
                            requires_auth=server_config.get("requires_auth", False),
                            config=server_config.get("config", {})
                        )

                logger.info(f"Discovered {len(config)} additional MCP servers")

        except Exception as e:
            logger.warning(f"Could not discover MCP servers: {e}")

    async def health_check(self, server_name: str) -> bool:
        """Check if an MCP server is available"""
        if server_name not in self.servers:
            return False

        server = self.servers[server_name]

        try:
            # TODO: Implement actual health check
            # For now, simulate with availability flag
            import time
            server.last_health_check = time.time()
            return server.is_available

        except Exception as e:
            logger.error(f"Health check failed for {server_name}: {e}")
            server.is_available = False
            return False

    def get_servers_by_capability(
        self,
        capability: MCPCapability
    ) -> List[MCPServer]:
        """Get all servers that support a specific capability"""

        servers = []
        for server in self.servers.values():
            if capability in server.capabilities and server.is_available:
                servers.append(server)

        # Sort by performance (latency and success rate)
        servers.sort(
            key=lambda s: s.avg_latency_ms / s.success_rate
        )

        return servers

    def get_best_server(
        self,
        capability: MCPCapability,
        prefer_local: bool = True
    ) -> Optional[MCPServer]:
        """Get the best server for a specific capability"""

        candidates = self.get_servers_by_capability(capability)

        if not candidates:
            return None

        if prefer_local:
            # Prefer stdio protocol (local) over network
            local_servers = [s for s in candidates if s.protocol == "stdio"]
            if local_servers:
                return local_servers[0]

        # Return best performing server
        return candidates[0]

    def register_server(self, server: MCPServer):
        """Register a new MCP server"""
        self.servers[server.name] = server
        logger.info(f"Registered MCP server: {server.name}")

    def unregister_server(self, server_name: str):
        """Unregister an MCP server"""
        if server_name in self.servers:
            del self.servers[server_name]
            logger.info(f"Unregistered MCP server: {server_name}")


class MCPTaskRouter:
    """Routes tasks to appropriate MCP servers based on requirements"""

    def __init__(self, registry: MCPServerRegistry):
        self.registry = registry

        # Task patterns to capability mapping
        self.task_patterns = {
            # File operations
            r"(read|write|create|delete|modify).*file": MCPCapability.FILE_OPERATIONS,
            r"(list|browse).*directory": MCPCapability.FILE_OPERATIONS,

            # Code execution
            r"(run|execute|eval).*code": MCPCapability.CODE_EXECUTION,
            r"(python|javascript|typescript).*script": MCPCapability.CODE_EXECUTION,

            # Web search
            r"(search|find|look up).*web": MCPCapability.WEB_SEARCH,
            r"(google|bing|duckduckgo)": MCPCapability.WEB_SEARCH,

            # GitHub
            r"(github|pull request|issue|repository)": MCPCapability.GITHUB_INTEGRATION,
            r"(commit|push|merge|branch)": MCPCapability.GITHUB_INTEGRATION,

            # Browser automation
            r"(screenshot|scrape|automate).*browser": MCPCapability.BROWSER_AUTOMATION,
            r"(click|fill|navigate).*page": MCPCapability.BROWSER_AUTOMATION,

            # Data analysis
            r"(analyze|process|visualize).*data": MCPCapability.DATA_ANALYSIS,
            r"(pandas|numpy|matplotlib)": MCPCapability.DATA_ANALYSIS,

            # Database
            r"(query|select|insert|update).*database": MCPCapability.DATABASE_QUERIES,
            r"(sql|mongodb|redis)": MCPCapability.DATABASE_QUERIES,

            # Docker
            r"(docker|container|image)": MCPCapability.DOCKER_MANAGEMENT,
            r"(build|deploy|scale).*container": MCPCapability.DOCKER_MANAGEMENT,

            # Security
            r"(scan|audit|check).*security": MCPCapability.SECURITY_SCANNING,
            r"(vulnerability|exploit|cve)": MCPCapability.SECURITY_SCANNING,
        }

    def identify_required_capability(self, task: str) -> Optional[MCPCapability]:
        """Identify which capability is needed for a task"""

        import re

        task_lower = task.lower()

        for pattern, capability in self.task_patterns.items():
            if re.search(pattern, task_lower):
                return capability

        return None

    def select_server(
        self,
        task: str,
        preferred_servers: List[str] = None
    ) -> Optional[MCPServer]:
        """Select the best MCP server for a task"""

        # Identify required capability
        capability = self.identify_required_capability(task)

        if not capability:
            logger.warning(f"Could not identify capability for task: {task}")
            return None

        # Check preferred servers first
        if preferred_servers:
            for server_name in preferred_servers:
                if server_name in self.registry.servers:
                    server = self.registry.servers[server_name]
                    if capability in server.capabilities and server.is_available:
                        return server

        # Get best available server
        return self.registry.get_best_server(capability)

    def select_multiple_servers(
        self,
        tasks: List[str]
    ) -> Dict[str, MCPServer]:
        """Select servers for multiple tasks"""

        selections = {}

        for task in tasks:
            server = self.select_server(task)
            if server:
                selections[task] = server

        return selections

    async def route_task(
        self,
        task: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Route a task to the appropriate MCP server and execute"""

        server = self.select_server(task)

        if not server:
            return {
                "success": False,
                "error": "No suitable MCP server found for task",
                "task": task
            }

        # Execute task on selected server
        try:
            result = await self._execute_on_server(server, task, **kwargs)
            return result

        except Exception as e:
            logger.error(f"Task execution failed on {server.name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "task": task,
                "server": server.name
            }

    async def _execute_on_server(
        self,
        server: MCPServer,
        task: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute task on specific MCP server"""

        # TODO: Implement actual MCP protocol communication
        # For now, return mock result

        await asyncio.sleep(server.avg_latency_ms / 1000)

        return {
            "success": True,
            "result": f"Task executed on {server.name}",
            "task": task,
            "server": server.name,
            "latency_ms": server.avg_latency_ms
        }


class MCPServerOptimizer:
    """Optimizes MCP server selection based on performance history"""

    def __init__(self, registry: MCPServerRegistry):
        self.registry = registry
        self.performance_history = {}

    def update_metrics(
        self,
        server_name: str,
        success: bool,
        latency_ms: float
    ):
        """Update server performance metrics"""

        if server_name not in self.registry.servers:
            return

        server = self.registry.servers[server_name]

        # Update running averages
        alpha = 0.1  # Exponential moving average factor

        server.avg_latency_ms = (1 - alpha) * server.avg_latency_ms + alpha * latency_ms

        if success:
            server.success_rate = (1 - alpha) * server.success_rate + alpha
        else:
            server.success_rate = (1 - alpha) * server.success_rate

        # Store in history
        if server_name not in self.performance_history:
            self.performance_history[server_name] = []

        self.performance_history[server_name].append({
            "timestamp": time.time(),
            "success": success,
            "latency_ms": latency_ms
        })

        # Keep only recent history (last 100 entries)
        self.performance_history[server_name] = self.performance_history[server_name][-100:]

    def get_recommendations(
        self,
        capability: MCPCapability
    ) -> List[Dict[str, Any]]:
        """Get recommendations for improving server performance"""

        recommendations = []
        servers = self.registry.get_servers_by_capability(capability)

        for server in servers:
            if server.success_rate < 0.9:
                recommendations.append({
                    "server": server.name,
                    "issue": "low_success_rate",
                    "current": server.success_rate,
                    "recommendation": "Consider fallback server or retry logic"
                })

            if server.avg_latency_ms > 1000:
                recommendations.append({
                    "server": server.name,
                    "issue": "high_latency",
                    "current": server.avg_latency_ms,
                    "recommendation": "Consider caching or local alternative"
                })

        return recommendations