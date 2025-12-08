
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv

load_dotenv()

from src.server import CognitiveWorkspace, CWDConfig


def test_deconstruct_persistence():
    print("Initializing CognitiveWorkspace...")
    try:
        config = CWDConfig()
        # Use temp dir for Chroma to check logic without touching possibly corrupted main DB
        import tempfile
        config.chroma_path = tempfile.mkdtemp()
        print(f"Using temp Chroma path: {config.chroma_path}")

        workspace = CognitiveWorkspace(config)

        # Inject Manifold (Required for deconstruct)
        from src.manifold import HopfieldConfig, Manifold
        hopfield_cfg = HopfieldConfig(embedding_dim=1024) # dummy dim
        workspace.manifold = Manifold(hopfield_cfg)
        print("Injected Manifold into Workspace")
    except Exception as e:
        print(f"Failed to initialize workspace: {e}")
        return

    problem = "Why is the sky blue?"
    print(f"Deconstructing: {problem}")

    try:
        # Call deconstruct
        result = workspace.deconstruct(problem)

        print("\nDECONSTRUCTION RESULT:")
        print(f"Root ID: {result.get('root_id')}")
        print(f"Components: {len(result.get('components', []))}")
        for comp in result.get('components', []):
            print(f"- [{comp.get('type')}] {comp.get('content')[:50]}...")

        embeddings = result.get('embeddings', {})
        print(f"Embeddings: {list(embeddings.keys())}")

        # Verify Persistence in Neo4j
        print("\nVERIFYING NEO4J PERSISTENCE...")
        with workspace.neo4j_driver.session() as session:
            # Check for root node
            root = session.run("MATCH (n:ThoughtNode {id: $id}) RETURN n", id=result['root_id']).single()
            if root:
                print(f"[PASS] Root node found: {root['n']['cognitive_type']}")
            else:
                print(f"[FAIL] Root node NOT found!")

            # Check for component nodes
            for comp in result['components']:
                cid = comp['id']
                node = session.run("MATCH (n:ThoughtNode {id: $id}) RETURN n", id=cid).single()
                if node:
                    print(f"[PASS] Component {comp['type']} found: {node['n']['cognitive_type']}")
                else:
                    print(f"[FAIL] Component {cid} NOT found!")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        workspace.close()

if __name__ == "__main__":
    test_deconstruct_persistence()
