"""
Emotion Evolution Framework: Computational Empathy for Reflective Agent Architecture

This module loads and provides query access to the comprehensive emotion evolution
framework, enabling empathic understanding and evolutionarily-grounded emotional
intelligence in AI interactions.

The framework provides:
- Basic emotion models (fear, anger, disgust, joy, sadness, surprise)
- Valence-Arousal mapping compatible with AffectVector
- Evolutionary layers (homeostatic → tertiary process)
- AI interaction guidelines and ethical considerations
- Computational Empathy Architecture for value integration
- Empathic response templates

Integration with existing RAA components:
- Grok-Lang (grok_lang.py): Uses AffectVector for empathic alignment
- Director: Can use emotional context for goal reframing
- Advisors: Evolutionary Psychologist profile references this framework
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmotionEvolutionFramework:
    """
    Loader and query interface for the Emotion Evolution Framework.

    Provides structured access to:
    - Basic and complex emotions with neural correlates
    - Valence-Arousal emotional mapping
    - Evolutionary layers of emotional processing
    - AI interaction guidelines
    - Computational Empathy Architecture
    - Empathic response templates
    """

    # Basic emotions as defined in the framework
    BASIC_EMOTIONS = ["fear", "anger", "disgust", "joy_happiness", "sadness", "surprise"]

    # Valence-Arousal quadrant mappings
    VALENCE_AROUSAL_MAP = {
        "high_arousal_positive": ["excitement", "joy", "elation"],
        "high_arousal_negative": ["fear", "anger", "panic"],
        "low_arousal_positive": ["contentment", "serenity", "love"],
        "low_arousal_negative": ["sadness", "depression", "boredom"]
    }

    def __init__(self, framework_path: Optional[str] = None):
        """
        Load the emotion evolution framework.

        Args:
            framework_path: Path to the JSON file. Defaults to src/config/emotion_evolution_framework.json
        """
        if framework_path is None:
            # Default path relative to this module
            module_dir = Path(__file__).parent.parent
            framework_path = module_dir / "config" / "emotion_evolution_framework.json"
        else:
            framework_path = Path(framework_path)

        self.framework_path = framework_path
        self._data: Dict[str, Any] = {}
        self._load_framework()

    def _load_framework(self) -> None:
        """Load the framework JSON file."""
        try:
            with open(self.framework_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                # The actual data is nested under 'data' key
                self._data = raw.get("data", raw)
                self._metadata = {
                    "name": raw.get("name", "emotion_evolution_framework"),
                    "domain": raw.get("domain", ""),
                    "description": raw.get("description", ""),
                    "tags": raw.get("tags", [])
                }
            logger.info(f"Loaded Emotion Evolution Framework from {self.framework_path}")
        except FileNotFoundError:
            logger.warning(f"Emotion Evolution Framework not found at {self.framework_path}")
            self._data = {}
            self._metadata = {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Emotion Evolution Framework: {e}")
            self._data = {}
            self._metadata = {}

    @property
    def is_loaded(self) -> bool:
        """Check if the framework was successfully loaded."""
        return bool(self._data)

    def get_basic_emotion(self, emotion_name: str) -> Optional[Dict[str, Any]]:
        """
        Get full details for a basic emotion.

        Args:
            emotion_name: One of: fear, anger, disgust, joy_happiness, sadness, surprise

        Returns:
            Dict with adaptive_function, neural_circuit, behavioral_output,
            physiological, universal_expression, variants
        """
        emotions = self._data.get("functional_architecture", {}).get(
            "basic_emotion_theory", {}
        ).get("core_emotions", {})

        # Handle aliases
        if emotion_name.lower() in ["joy", "happiness"]:
            emotion_name = "joy_happiness"

        return emotions.get(emotion_name.lower())

    def get_complex_emotion(self, emotion_name: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a complex emotion (guilt, pride, jealousy, romantic_love).

        Args:
            emotion_name: One of: guilt, pride, jealousy, romantic_love

        Returns:
            Dict with components, cognitive_appraisal, social_function, etc.
        """
        complex_emotions = self._data.get("functional_architecture", {}).get(
            "basic_emotion_theory", {}
        ).get("complex_emotions", {}).get("examples", {})

        return complex_emotions.get(emotion_name.lower())

    def get_evolutionary_layer(self, layer: int) -> Optional[Dict[str, Any]]:
        """
        Get details for an evolutionary layer (1-4).

        Args:
            layer: 1 (homeostatic), 2 (primary_process), 3 (secondary_process), 4 (tertiary_process)

        Returns:
            Dict with emergence, characteristics, neural_substrate, acip_level
        """
        layer_map = {
            1: "layer_1_homeostatic",
            2: "layer_2_primary_process",
            3: "layer_3_secondary_process",
            4: "layer_4_tertiary_process"
        }

        timeline = self._data.get("evolutionary_foundations", {}).get(
            "evolutionary_timeline", {}
        )

        layer_key = layer_map.get(layer)
        return timeline.get(layer_key) if layer_key else None

    def get_ai_guidelines(self) -> Dict[str, Any]:
        """
        Get AI interaction guidelines and ethical considerations.

        Returns:
            Dict with key_understanding_principles, interaction_guidelines,
            design_considerations, ethical_considerations
        """
        return self._data.get("implications_for_ai", {})

    def get_ai_principles(self) -> List[str]:
        """
        Get the 7 key understanding principles for AI systems.

        Returns:
            List of principle strings
        """
        principles = self._data.get("implications_for_ai", {}).get(
            "key_understanding_principles", {}
        )
        return [v for k, v in sorted(principles.items())]

    def get_empathic_template(self, context: str) -> Optional[str]:
        """
        Get empathic response template for a given emotional context.

        Args:
            context: One of: distress, joy, anxiety

        Returns:
            Template string for empathic response
        """
        templates = self._data.get("practical_applications", {}).get(
            "empathic_responses", {}
        ).get("templates", {})

        return templates.get(context.lower())

    def get_computational_empathy_architecture(self) -> Dict[str, Any]:
        """
        Get the Computational Empathy Architecture section.

        This describes the unified architecture for consciousness modeling
        and ethical alignment through Value Vectors.

        Returns:
            Dict with core_thesis, value_integration_framework, operational_implications
        """
        return self._data.get("computational_empathy_architecture", {})

    def map_affect_to_emotions(
        self,
        valence: float,
        arousal: float
    ) -> List[str]:
        """
        Map VAD values to likely emotions using the valence-arousal framework.

        Compatible with AffectVector from grok_lang.py.

        Args:
            valence: -1 (negative) to +1 (positive)
            arousal: 0 (calm) to 1 (activated)

        Returns:
            List of likely emotion names for this affective state
        """
        # Determine quadrant
        is_positive = valence > 0
        is_high_arousal = arousal > 0.5

        if is_positive and is_high_arousal:
            quadrant = "high_arousal_positive"
        elif not is_positive and is_high_arousal:
            quadrant = "high_arousal_negative"
        elif is_positive and not is_high_arousal:
            quadrant = "low_arousal_positive"
        else:
            quadrant = "low_arousal_negative"

        # Get from framework data
        mapping = self._data.get("functional_architecture", {}).get(
            "valence_arousal_framework", {}
        ).get("mapping", self.VALENCE_AROUSAL_MAP)

        return mapping.get(quadrant, [])

    def get_neurobiological_architecture(self) -> Dict[str, Any]:
        """
        Get the neurobiological architecture section.

        Returns:
            Dict with hierarchical_processing, key_circuits, neurochemical_systems
        """
        return self._data.get("neurobiological_architecture", {})

    def get_acip_integration(self) -> Dict[str, Any]:
        """
        Get the ACIP (Awareness Continuum) integration mapping.

        Shows how emotions map to consciousness layers.

        Returns:
            Dict with mapping_to_continuum and key_insights
        """
        return self._data.get("acip_integration", {})

    def get_emotional_regulation_strategies(self) -> Dict[str, Any]:
        """
        Get emotional regulation strategies (reappraisal, suppression, etc.).

        Returns:
            Dict of regulation strategies with descriptions and effectiveness
        """
        return self._data.get("functional_architecture", {}).get(
            "cognitive_emotional_integration", {}
        ).get("cognition_regulates_emotion", {})

    def query(self, query_type: str, query_param: str = "") -> Dict[str, Any]:
        """
        Unified query interface for the framework.

        Args:
            query_type: One of:
                - "basic_emotion": Get details for a basic emotion
                - "complex_emotion": Get details for a complex emotion
                - "evolutionary_layer": Get evolutionary layer details (1-4)
                - "ai_guidelines": Get AI interaction guidelines
                - "ai_principles": Get the 7 key AI principles
                - "empathic_template": Get empathic response template
                - "computational_empathy": Get Computational Empathy Architecture
                - "affect_mapping": Map valence,arousal to emotions
                - "neurobiology": Get neurobiological architecture
                - "acip": Get ACIP consciousness integration
                - "regulation": Get emotional regulation strategies

            query_param: Parameter for the query (emotion name, layer number, etc.)

        Returns:
            Query result as dict
        """
        if not self.is_loaded:
            return {"error": "Framework not loaded", "loaded": False}

        result: Any = None

        if query_type == "basic_emotion":
            result = self.get_basic_emotion(query_param)
        elif query_type == "complex_emotion":
            result = self.get_complex_emotion(query_param)
        elif query_type == "evolutionary_layer":
            try:
                layer = int(query_param)
                result = self.get_evolutionary_layer(layer)
            except ValueError:
                return {"error": f"Invalid layer number: {query_param}"}
        elif query_type == "ai_guidelines":
            result = self.get_ai_guidelines()
        elif query_type == "ai_principles":
            result = {"principles": self.get_ai_principles()}
        elif query_type == "empathic_template":
            template = self.get_empathic_template(query_param)
            result = {"context": query_param, "template": template} if template else None
        elif query_type == "computational_empathy":
            result = self.get_computational_empathy_architecture()
        elif query_type == "affect_mapping":
            # Expected format: "valence,arousal" e.g. "-0.5,0.8"
            try:
                parts = query_param.split(",")
                valence = float(parts[0].strip())
                arousal = float(parts[1].strip()) if len(parts) > 1 else 0.5
                emotions = self.map_affect_to_emotions(valence, arousal)
                result = {"valence": valence, "arousal": arousal, "likely_emotions": emotions}
            except (ValueError, IndexError):
                return {"error": f"Invalid affect format: {query_param}. Use 'valence,arousal'"}
        elif query_type == "neurobiology":
            result = self.get_neurobiological_architecture()
        elif query_type == "acip":
            result = self.get_acip_integration()
        elif query_type == "regulation":
            result = self.get_emotional_regulation_strategies()
        else:
            return {
                "error": f"Unknown query type: {query_type}",
                "valid_types": [
                    "basic_emotion", "complex_emotion", "evolutionary_layer",
                    "ai_guidelines", "ai_principles", "empathic_template",
                    "computational_empathy", "affect_mapping", "neurobiology",
                    "acip", "regulation"
                ]
            }

        if result is None:
            return {"error": f"Not found: {query_param} in {query_type}"}

        return {"query_type": query_type, "result": result}


# Global singleton instance for easy access
_framework_instance: Optional[EmotionEvolutionFramework] = None


def get_emotion_framework() -> EmotionEvolutionFramework:
    """Get or create the global EmotionEvolutionFramework instance."""
    global _framework_instance
    if _framework_instance is None:
        _framework_instance = EmotionEvolutionFramework()
    return _framework_instance


def consult_computational_empathy(
    query_type: str,
    query_param: str = ""
) -> Dict[str, Any]:
    """
    Consult the Emotion Evolution Framework for computational empathy insights.

    This is the function that implements the 'consult_computational_empathy' tool
    referenced by the Evolutionary Psychologist advisor.

    Args:
        query_type: Type of query (basic_emotion, ai_guidelines, empathic_template, etc.)
        query_param: Optional parameter for the query

    Returns:
        Query result from the framework
    """
    framework = get_emotion_framework()
    return framework.query(query_type, query_param)
