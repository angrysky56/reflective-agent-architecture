"""
Integration: Full Reflective Agent Architecture Loop

Integrates all four components into the complete "Aha!" loop:

1. Task Input → Pointer sets initial goal
2. Processor generates response (biased by goal)
3. Director monitors entropy → Detects "clash"
4. Director suppresses current goal
5. Director searches Manifold for alternative basin
6. Director updates Pointer with new goal
7. Processor resumes with new framing → Success
"""

from .raa_loop import RAAConfig, ReflectiveAgentArchitecture

__all__ = ["ReflectiveAgentArchitecture", "RAAConfig"]
