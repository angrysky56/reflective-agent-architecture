"""Seed script for Geometric Value Theory (GVT) nodes in Neo4j.

Creates foundational theory and axiom nodes that establish the geometric
framework for ethical reasoning within the Reflective Agent Architecture.

Usage:
    uv run python -m reflective_agent_architecture.scripts.seed_gvt
"""

import asyncio

from reflective_agent_architecture.config.cwd_config import CWDConfig


async def seed_gvt() -> None:
    """
    Seed the Neo4j database with Geometric Value Theory nodes.

    Creates:
        - Core GVT theory node (authored by 'ontologist' advisor)
        - Three foundational axioms: Curvature, Dimensionality, Invariants
    """
    print("Initializing Workspace...")
    settings = CWDConfig()
    # Use Neo4j driver directly for this seed script (avoids workspace dependency overhead)
    from neo4j import GraphDatabase

    uri = settings.neo4j_uri
    user = settings.neo4j_user
    password = (
        settings.neo4j_password.get_secret_value()
        if hasattr(settings.neo4j_password, "get_secret_value")
        else settings.neo4j_password
    )

    print(f"Connecting to Neo4j at {uri}...")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        print("Creating 'Geometric Value Theory' Nodes...")

        # 1. Create the Main Theory Node
        session.run(
            """
            MERGE (t:ThoughtNode {id: 'thought_gvt_core'})
            SET t.name = 'Geometric Value Theory',
                t.content = 'Ethics is not a code of conduct, but the Geometry of the Action Space. Values act as curvature (metric tensor) determining the geodesic (path of least resistance).',
                t.type = 'Theory',
                t.status = 'crystallized',
                t.created_at = timestamp()

            MERGE (a:Advisor {id: 'ontologist'})
            MERGE (a)-[:AUTHORED]->(t)
        """
        )

        # 2. Create the Axioms
        axioms = [
            (
                "thought_gvt_axiom_1",
                "Axiom of Curvature",
                "Values act like Mass in General Relativity, bending the metric of the decision space.",
            ),
            (
                "thought_gvt_axiom_2",
                "Axiom of Dimensionality",
                "Ethical maturity is an increase in manifold dimensions (R^2 -> R^n).",
            ),
            (
                "thought_gvt_axiom_3",
                "Axiom of Invariants",
                "Core values are Topological Invariants that persist under deformation of the social fabric.",
            ),
        ]

        for pid, name, content in axioms:
            session.run(
                """
                MERGE (p:ThoughtNode {id: $pid})
                SET p.name = $name,
                    p.content = $content,
                    p.type = 'Axiom',
                    p.status = 'crystallized',
                    p.created_at = timestamp()

                WITH p
                MATCH (t:ThoughtNode {id: 'thought_gvt_core'})
                MERGE (t)-[:CONTAINS]->(p)
            """,
                parameters={"pid": pid, "name": name, "content": content},
            )

        print("Success! Geometric Value Theory has been crystallized.")

    driver.close()


if __name__ == "__main__":
    asyncio.run(seed_gvt())
