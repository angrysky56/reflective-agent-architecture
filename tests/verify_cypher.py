import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

query = """
MATCH (n:ConceptNode)
OPTIONAL MATCH (n)--(m)
WITH n, count(m) as degree
WHERE degree < 2
RETURN n.id as id, n.name as name, n.content as content
LIMIT 5
"""

print(f"Testing query:\n{query}")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run(query)
        records = [r for r in result]
        print(f"SUCCESS: Query executed. Found {len(records)} records.")
    driver.close()
except Exception as e:
    print(f"FAILURE: Query failed: {e}")
