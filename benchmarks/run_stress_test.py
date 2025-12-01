import asyncio
import json
import logging
import os
from datetime import datetime

from src.compass.adapters import RAALLMProvider
from src.compass.compass_framework import create_compass
from src.compass.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CognitiveStressTest")

async def run_benchmarks():
    print("\n=== Starting Cognitive Stress Test (Inspired by Humanity's Last Exam) ===\n")

    # Initialize Configuration and Provider
    config = get_config()
    llm_provider = RAALLMProvider(model_name=config.intelligence.llm_model)

    # Initialize COMPASS with standard RAA Provider
    compass = create_compass(config=config, llm_provider=llm_provider)

    # Load Questions
    questions_path = os.path.join(os.path.dirname(__file__), "questions.json")
    with open(questions_path, 'r') as f:
        questions = json.load(f)

    results = []

    for i, q in enumerate(questions):
        print(f"\n--- Question {i+1}/{len(questions)}: {q['category']} ({q['difficulty']}) ---")
        print(f"Q: {q['question']}")

        start_time = datetime.now()

        # Execute Task via COMPASS
        # We use process_task which triggers the full cognitive loop
        try:
            response = await compass.process_task(q['question'])
            output = response.get("final_report") or response.get("solution", "No result returned")
            status = "Success"
        except Exception as e:
            logger.error(f"Error processing question {q['id']}: {e}")
            output = str(e)
            status = "Error"

        duration = (datetime.now() - start_time).total_seconds()

        print(f"A: {output[:200]}...") # Print preview
        print(f"Time: {duration:.2f}s")

        results.append({
            "id": q['id'],
            "category": q['category'],
            "question": q['question'],
            "answer": output,
            "status": status,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })

    # Save Results
    output_path = os.path.join(os.path.dirname(__file__), f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== Stress Test Complete ===")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(run_benchmarks())
