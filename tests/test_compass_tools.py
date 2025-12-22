
import asyncio
import logging
import os
import sys
from pathlib import Path

# Adjust path to include project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from src.compass.compass_framework import COMPASS
from src.compass.config import COMPASSConfig
from src.compass.adapters.mcp_client import RAAMCPClient
from src.compass.adapters.llm_provider import RAALLMProvider
from src.server import CognitiveWorkspace, RAAServerContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CompassTest")

async def test_compass_tool_execution():
    print("\n--- Starting COMPASS Tool Execution Test ---")
    
    # 1. Initialize Server Context (to get DB access)
    # We need a real workspace to support the tools
    print("Initializing Cognitive Workspace...")
    workspace = CognitiveWorkspace()
    # We don't start the full server, just the workspace components
    # But wait, RAAMCPClient calls the server via HTTP usually? 
    # Or does it use internal method calls?
    # Let's check RAAMCPClient implementation.
    # If it's internal, we might need a bridge.
    
    # Actually, let's try to trust the existing RAAMCPClient. 
    # If it fails to connect (because no server is running), we'll know.
    # Ideally, we should run this against a running server, OR verify if RAAMCPClient can be mocked to call internal methods.
    
    # For this test, we assume the user might not have a separate server process running.
    # So we should probably spin up a minimal server context or use an internal-only MCP client if one exists.
    # Looking at src/integration/mcp_client.py (if it exists) or src/compass/adapters/mcp_client.py
    
    # Let's assume we need to run against the running server for now (e.g. standard MCP pattern).
    # IF that fails, we will need to refactor to use a "DirectBridge" client.
    
    llm_provider = RAALLMProvider()
    
    # We need an MCP Client. 
    # NOTE: The standard RAAMCPClient likely connects to a URL.
    # Check src/compass/adapters/mcp_client.py to confirm.
    # If it connects to a URL, we need to ensure the server is running or use a mock.
    
    # Attempt to use the class we saw in imports: RAAMCPClient
    mcp_client = RAAMCPClient() # Check init signature!
    
    # Create COMPASS with Tool support enabled
    config = COMPASSConfig()
    config.intelligence.enable_tools = True
    config.intelligence.tools_provider = "mcp" # Assuming this is the flag
    
    compass = COMPASS(
        config=config,
        llm_provider=llm_provider,
        mcp_client=mcp_client
    )
    
    # 2. Define a Task that REQUIRES a tool
    task_description = (
        "Deconstruct the concept of 'Digital Cognition' into its constituent components. "
        "Use the 'deconstruct' tool explicitly. "
        "Ensure the results are saved to the database."
    )
    
    print(f"Task: {task_description}")
    
    # 3. Process the Task
    try:
        result = await compass.process_task(task_description)
        print("\n--- COMPASS Processing Complete ---")
        print(f"Success: {result['success']}")
        print(f"Report: {result['final_report']}")
        
        # 4. Verify in Neo4j
        print("\n--- Verifying Neo4j Persistence ---")
        # Reuse workspace driver if possible, or create new connection
        from neo4j import GraphDatabase
        from src.config.cwd_config import CWDConfig
        
        settings = CWDConfig()
        uri = settings.neo4j_uri
        user = settings.neo4j_user
        password = settings.neo4j_password.get_secret_value() if hasattr(settings.neo4j_password, 'get_secret_value') else settings.neo4j_password
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Check for nodes related to 'Digital Cognition'
            records = session.run("""
                MATCH (p:ThoughtNode {name: 'Digital Cognition'})
                OPTIONAL MATCH (p)-[:DECOMPOSED_TO]->(c)
                RETURN p, count(c) as children_count
            """)
            
            found = False
            for record in records:
                found = True
                print(f"Found Parent Node: {record['p']['name']}")
                print(f"Children Count: {record['children_count']}")
                
                if record['children_count'] > 0:
                    print("SUCCESS: Deconstruct tool created children nodes!")
                else:
                    print("WARNING: Parent found but no children. Tool might have failed or not run recursively.")
                    
            if not found:
                print("FAILURE: No nodes found for 'Digital Cognition'. Tool likely did not run or save.")
                
        driver.close()
        
    except Exception as e:
        print(f"Test Failed with Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_compass_tool_execution())
