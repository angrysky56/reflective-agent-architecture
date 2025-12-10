import logging
from typing import Any, Dict

from src.compass.advisors.registry import AdvisorProfile, AdvisorRegistry

logger = logging.getLogger(__name__)


class AdvisorManager:
    """
    Integration component for managing Advisors.
    Bridging AdvisorRegistry (Config) with CognitiveWorkspace (Graph/Vector Store).

    Consolidates functionality for:
    - CRUD (Create, Read, Update, Delete)
    - Knowledge Management (Link/Unlink/Context)
    """

    def __init__(self, registry: AdvisorRegistry, workspace: Any = None):
        self.registry = registry
        self.workspace = workspace  # CognitiveWorkspace instance

    def manage_advisor(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified handler for all advisor management tasks.

        Args:
            action: One of ['create', 'update', 'delete', 'list', 'get', 'link_knowledge', 'get_knowledge', 'get_context']
            params: Parameters specific to the action

        Returns:
            Dict containing result or success status.
        """
        try:
            if action == "create" or action == "update":
                return self._create_or_update_advisor(params)
            elif action == "delete":
                return self._delete_advisor(params)
            elif action == "list":
                return self._list_advisors()
            elif action == "get":
                return self._get_advisor(params)
            elif action == "link_knowledge":
                return self._link_knowledge(params)
            elif action == "get_knowledge":
                return self._get_knowledge(params)
            elif action == "get_context":
                return self._get_context(params)
            else:
                return {"error": f"Unknown action: {action}"}
        except Exception as e:
            logger.error(f"AdvisorManager error ({action}): {e}", exc_info=True)
            return {"error": str(e)}

    def _create_or_update_advisor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update an advisor profile."""
        if "id" not in params or "name" not in params:
            return {"error": "Create requires 'id' and 'name'"}

        # Defaults
        profile = AdvisorProfile(
            id=params["id"],
            name=params["name"],
            role=params.get("role", "Advisor"),
            description=params.get("description", "A specialized advisor."),
            system_prompt=params.get("system_prompt", "You are a helpful advisor."),
            tools=params.get("tools", []),
            knowledge_node_ids=params.get("knowledge_node_ids", []),
        )

        self.registry.register_advisor(profile, save=True)
        return {
            "status": "success",
            "message": f"Advisor '{profile.name}' ({profile.id}) saved.",
            "advisor": profile.to_dict(),
        }

    def _delete_advisor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        advisor_id = params.get("advisor_id") or params.get("id")
        if not advisor_id:
            return {"error": "Delete requires 'advisor_id'"}

        if self.registry.remove_advisor(advisor_id):
            return {"status": "success", "message": f"Advisor '{advisor_id}' deleted."}
        else:
            return {"error": f"Advisor '{advisor_id}' not found."}

    def _list_advisors(self) -> Dict[str, Any]:
        advisors = [p.to_dict() for p in self.registry.get_all_advisors()]
        return {"count": len(advisors), "advisors": advisors}

    def _get_advisor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        advisor_id = params.get("advisor_id") or params.get("id")
        if not advisor_id:
            return {"error": "Get advisor requires 'advisor_id' or 'id'"}

        advisor = self.registry.get_advisor(advisor_id)
        if advisor:
            return {"advisor": advisor.to_dict()}
        return {"error": "Advisor not found"}

    def _link_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        advisor_id = params.get("advisor_id")
        node_id = params.get("node_id")

        if not advisor_id or not node_id:
            return {"error": "Linking requires 'advisor_id' and 'node_id'"}

        if self.registry.link_node_to_advisor(advisor_id, node_id):
            return {"status": "success", "message": f"Linked node {node_id} to {advisor_id}"}
        else:
            return {"error": "Failed to link. Check IDs."}

    def _get_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        advisor_id = params.get("advisor_id")
        if not advisor_id:
            return {"error": "Get knowledge requires 'advisor_id'"}
        return self.registry.get_advisor_knowledge(advisor_id)

    def _get_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve full text context from valid linked nodes."""
        advisor_id = params.get("advisor_id")
        if not advisor_id:
            return {"error": "Get context requires 'advisor_id'"}

        if not self.workspace:
            return {"error": "Workspace not available for context retrieval"}

        knowledge = self.registry.get_advisor_knowledge(advisor_id)
        node_ids = knowledge.get("node_ids", [])

        if not node_ids:
            return {"context": f"No knowledge linked to advisor '{advisor_id}'."}

        context_parts = [f"Knowledge Context for Advisor '{advisor_id}':"]

        # Access Neo4j/Chroma via workspace
        # We need to replicate server._get_contents_from_chroma logic effectively
        # Or expose it in workspace.

        try:
            # Use workspace's Neo4j session if available
            if hasattr(self.workspace, "neo4j_driver"):
                with self.workspace.neo4j_driver.session() as session:
                    # Fetch structure
                    records = session.run(
                        "MATCH (n:ThoughtNode) WHERE n.id IN $ids RETURN n.id as id, n.type as type",
                        ids=node_ids,
                    ).data()

                    found_ids = {r["id"] for r in records}

                    # Fetch content (try Chroma first via workspace collection)
                    content_map = {}
                    if hasattr(self.workspace, "collection") and self.workspace.collection:
                        try:
                            results = self.workspace.collection.get(ids=list(found_ids))
                            if results and results["ids"]:
                                for i, doc_id in enumerate(results["ids"]):
                                    content_map[doc_id] = results["documents"][i]
                        except Exception as e:
                            logger.warning(f"Chroma context fetch failed: {e}")

                    # Assemble context
                    for record in records:
                        nid = record["id"]
                        ntype = record.get("type", "thought")
                        content = content_map.get(nid, "[Content not found in vector store]")
                        context_parts.append(f"\n--- Node: {nid} ({ntype}) ---\n{content}")

            return {"context": "\n".join(context_parts)}

        except Exception as e:
            return {"error": f"Failed to retrieve context: {e}"}
