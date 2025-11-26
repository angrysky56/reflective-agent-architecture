"""
Sleep Cycle: Offline Learning & Consolidation

This module implements the "Night Mode" for the Reflective Agent.
It performs two key functions:
1. Replay (REM): Trains the Processor on high-quality (low energy) historical episodes.
2. Crystallization (Deep Sleep): Identifies frequent graph patterns and converts them into Tools.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.optim as optim

from src.persistence.work_history import WorkHistory
from src.processor.transformer_decoder import ProcessorConfig, TransformerDecoder

if TYPE_CHECKING:
    from src.server import CognitiveWorkspace

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class SleepCycle:
    def __init__(
        self,
        db_path: str = "raa_history.db",
        processor_config: Optional[ProcessorConfig] = None,
        workspace: Optional["CognitiveWorkspace"] = None
    ):
        self.history = WorkHistory(db_path)
        self.config = processor_config or ProcessorConfig()
        self.device = self.config.device

        # Initialize Processor
        self.processor = TransformerDecoder(self.config).to(self.device)
        self.optimizer = optim.AdamW(self.processor.parameters(), lr=1e-4)

        # Initialize Tokenizer (GPT-2 matches default vocab size 50257)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}. Using dummy encoding.")
            self.tokenizer = None

        self.workspace = workspace

    def dream(self, epochs: int = 1) -> Dict[str, Any]:
        """
        Execute the Sleep Cycle.
        """
        logger.info("Entering Sleep Cycle...")

        # 1. Replay (Training)
        replay_stats = self._replay_memories(epochs)

        # 2. Crystallization (Tool Creation)
        crystallization_stats = self._crystallize_patterns()

        return {
            "replay": replay_stats,
            "crystallization": crystallization_stats
        }

    def _replay_memories(self, epochs: int) -> Dict[str, Any]:
        """
        Train the processor on 'Focused' (high-quality) episodes.
        """
        logger.info("Replaying memories (SFT)...")

        # Fetch high-quality history
        # We want rows where cognitive_state is "Focused" or energy is low
        # Since we don't have a direct query for that in WorkHistory yet, we'll fetch all and filter
        # In a real system, this should be a SQL query

        # Mocking the fetch for now as WorkHistory.get_all_history() doesn't exist yet
        # But we can assume we can iterate over the DB or add a method
        # For this prototype, we will try to fetch from the DB connection directly if possible,
        # or fall back to simulation if DB is empty.

        episodes = []
        try:
            rows = self.history.get_focused_episodes(limit=100)
            for row in rows:
                # Format: "Operation: <op> Params: <params> Result: <res>"
                text = f"Operation: {row['operation']}\nParams: {row['params']}\nResult: {row['result_summary']}"
                episodes.append(text)
        except Exception as e:
            logger.warning(f"Failed to fetch history: {e}")

        if not episodes:
            logger.info("No focused memories found. Skipping replay.")
            return {"steps": 0, "avg_loss": 0.0}

        total_loss = 0.0
        steps = 0

        for epoch in range(epochs):
            for text in episodes:
                if not self.tokenizer:
                    break

                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    padding="max_length"
                )
                input_ids = inputs.input_ids.to(self.device)
                labels = input_ids.clone()

                # Train step
                loss = self.processor.train_step(input_ids, labels, self.optimizer)
                total_loss += loss
                steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0.0
        logger.info(f"Replay complete. Steps: {steps}, Avg Loss: {avg_loss:.4f}")

        return {"steps": steps, "avg_loss": avg_loss}

    def _crystallize_patterns(self) -> Dict[str, Any]:
        """
        Identify frequent patterns and convert to tools.
        """
        if not self.workspace:
            return {"status": "skipped", "reason": "No workspace connected"}

        logger.info("Crystallizing patterns (Tool Genesis)...")

        # Heuristic: Find nodes with high degree in Neo4j (simulated)
        # In production: query Neo4j for dense subgraphs

        # For prototype, we'll try to create a tool from a specific "dream" pattern
        # if we find enough related nodes.

        # This is a placeholder for the complex graph clustering logic
        return {"new_tools_created": 0, "message": "No dense clusters found (Prototype stub)"}

if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    sleep = SleepCycle()
    print(sleep.dream())
