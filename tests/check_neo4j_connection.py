import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to {uri} as {user}...")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"SUCCESS: Connected to Neo4j! Total nodes: {count}")
    driver.close()
except Exception as e:
    print(f"FAILURE: Could not connect to Neo4j: {e}")
