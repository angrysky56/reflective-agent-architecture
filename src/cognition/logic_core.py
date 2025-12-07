
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("raa.logic_core")

class SyntaxValidator:
    """Pre-validate logical formulas for common syntax errors"""

    # Common quantifiers and operators
    QUANTIFIERS = {"all", "exists"}
    OPERATORS = {"->", "<->", "&", "|", "-"}
    RESERVED = QUANTIFIERS | {"true", "false", "end_of_list"}

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate(self, formula: str) -> Tuple[bool, List[str], List[str]]:
        """Validate a logical formula"""
        self.errors = []
        self.warnings = []

        # Remove trailing period for analysis
        formula_clean = formula.rstrip(".")

        # Check balanced parentheses
        self._check_balanced_parens(formula_clean)

        # Check quantifier syntax
        self._check_quantifiers(formula_clean)

        # Check operator usage
        self._check_operators(formula_clean)

        # Check predicate/function naming
        self._check_naming(formula_clean)

        # Check for common mistakes
        self._check_common_mistakes(formula_clean)

        return (len(self.errors) == 0, self.errors, self.warnings)

    def _check_balanced_parens(self, formula: str):
        """Check if parentheses are balanced"""
        stack = []
        for i, char in enumerate(formula):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if not stack:
                    self.errors.append(f"Unmatched closing parenthesis at position {i}")
                else:
                    stack.pop()

        if stack:
            self.errors.append(f"Unmatched opening parenthesis at position {stack[0]}")

    def _check_quantifiers(self, formula: str):
        """Check quantifier syntax"""
        # Pattern: quantifier variable (formula)
        for quantifier in self.QUANTIFIERS:
            # Find all occurrences of this quantifier
            pattern = rf"\b{quantifier}\s+(\w+)"
            matches = re.finditer(pattern, formula)

            for match in matches:
                var = match.group(1)
                # Check if variable follows quantifier is lowercase
                if not var[0].islower():
                    self.warnings.append(f"Quantifier variable '{var}' should start with lowercase")

                # Check if there's a formula after the quantifier
                pos = match.end()
                remaining = formula[pos:].lstrip()
                if not remaining or remaining[0] != "(":
                    self.errors.append(f"Quantifier '{quantifier} {var}' must be followed by a formula in parentheses")

    def _check_operators(self, formula: str):
        """Check operator usage"""
        # Check for double operators (likely mistakes)
        for op in ["&", "|"]:
            if op + op in formula:
                self.warnings.append(f"Double operator '{op}{op}' found - did you mean to use it twice?")

        # Check for implication chains without parentheses
        if formula.count("->") > 1 and formula.count("(") == 0:
            self.warnings.append("Multiple implications without parentheses - consider adding parentheses for clarity")

    def _check_naming(self, formula: str):
        """Check predicate/function naming conventions"""
        # Extract potential predicate/function names
        # Pattern: word followed by opening paren
        pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        matches = re.finditer(pattern, formula)

        for match in matches:
            name = match.group(1)

            # Skip quantifiers
            if name in self.QUANTIFIERS:
                continue

            # Predicates should start with lowercase
            if name[0].isupper():
                self.warnings.append(f"Predicate/function '{name}' starts with uppercase - consider using lowercase for consistency")

            # Check for reserved words
            if name in self.RESERVED:
                self.errors.append(f"'{name}' is a reserved keyword and cannot be used as a predicate/function")

    def _check_common_mistakes(self, formula: str):
        """Check for common syntax mistakes"""
        # Missing spaces around operators
        for op in ["->", "<->"]:
            # Check for operators without spaces
            pattern = rf"\w{re.escape(op)}\w"
            if re.search(pattern, formula):
                self.warnings.append(f"Consider adding spaces around '{op}' for readability")

        # Unquoted strings (should be predicates)
        if '"' in formula or "'" in formula:
            self.warnings.append("Strings in quotes are not standard in first-order logic - use predicates or constants instead")

        # Empty parentheses
        if "()" in formula:
            self.errors.append("Empty parentheses found - predicates and functions must have arguments")

class CategoricalHelpers:
    """Helper functions for categorical reasoning in first-order logic"""

    @staticmethod
    def category_axioms() -> List[str]:
        """Generate basic category theory axioms"""
        return [
            # Identity morphisms exist
            "all x (object(x) -> exists i (morphism(i) & source(i,x) & target(i,x) & identity(i,x)))",
            # Identity is unique
            "all x all i1 all i2 ((identity(i1,x) & identity(i2,x)) -> i1 = i2)",
            # Composition exists when source/target match
            "all f all g ((morphism(f) & morphism(g) & target(f) = source(g)) -> exists h (morphism(h) & compose(g,f,h)))",
            # Composition is associative
            "all f all g all h all fg all gh all fgh all gfh ((compose(g,f,fg) & compose(h,g,gh) & compose(h,fg,fgh) & compose(gh,f,gfh)) -> fgh = gfh)",
            # Left identity law
            "all f all a all id ((morphism(f) & source(f,a) & identity(id,a) & compose(f,id,comp)) -> comp = f)",
            # Right identity law
            "all f all b all id ((morphism(f) & target(f,b) & identity(id,b) & compose(id,f,comp)) -> comp = f)",
        ]

    @staticmethod
    def functor_axioms(functor_name: str = "F") -> List[str]:
        """Generate functor axioms"""
        f = functor_name.lower()
        return [
            # Functor preserves identity
            f"all x all id (identity(id,x) -> identity({f}(id), {f}(x)))",
            # Functor preserves composition
            f"all g all h all gh ((compose(g,h,gh)) -> compose({f}(g), {f}(h), {f}(gh)))",
        ]

    @staticmethod
    def verify_commutativity(path_a: List[str], path_b: List[str], object_start: str, object_end: str) -> Tuple[List[str], str]:
        """Generate FOL to verify diagram commutativity"""
        premises = []

        # Define morphisms in path A
        for i, morph in enumerate(path_a):
            if i == 0:
                premises.append(f"morphism({morph})")
                premises.append(f"source({morph}, {object_start})")
            else:
                premises.append(f"morphism({morph})")

            if i == len(path_a) - 1:
                premises.append(f"target({morph}, {object_end})")

        # Define morphisms in path B
        for i, morph in enumerate(path_b):
            if i == 0:
                premises.append(f"morphism({morph})")
                premises.append(f"source({morph}, {object_start})")
            else:
                premises.append(f"morphism({morph})")

            if i == len(path_b) - 1:
                premises.append(f"target({morph}, {object_end})")

        # Compose paths
        comp_a = CategoricalHelpers._compose_path_helper(path_a, "comp_a")
        comp_b = CategoricalHelpers._compose_path_helper(path_b, "comp_b")

        premises.extend(comp_a["premises"])
        premises.extend(comp_b["premises"])

        # Conclusion: composed paths are equal
        conclusion = f"{comp_a['result']} = {comp_b['result']}"

        return (premises, conclusion)

    @staticmethod
    def natural_transformation_condition(functor_f: str = "F", functor_g: str = "G", component: str = "alpha") -> List[str]:
        """Generate naturality condition for a natural transformation"""
        f_lower = functor_f.lower()
        g_lower = functor_g.lower()

        return [
            # Naturality condition
            f"all morph all a all b ((morphism(morph) & source(morph,a) & target(morph,b)) -> exists comp1 exists comp2 (compose({g_lower}(morph), {component}(a), comp1) & compose({component}(b), {f_lower}(morph), comp2) & comp1 = comp2))"
        ]

    @staticmethod
    def _compose_path_helper(path: List[str], result_name: str) -> Dict:
        """Helper to generate composition premises for a path"""
        if len(path) == 1:
            return {"premises": [], "result": path[0]}

        premises = []
        current = path[0]

        for i in range(1, len(path)):
            temp_name = f"{result_name}_temp_{i}" if i < len(path) - 1 else result_name
            premises.append(f"compose({path[i]}, {current}, {temp_name})")
            current = temp_name

        return {"premises": premises, "result": current}

    @staticmethod
    def monoid_axioms() -> List[str]:
        """Axioms for a monoid"""
        return [
            "all x all y exists z (mult(x,y,z))",
            "all x all y all z all xy all yz all xyz all ybc ((mult(x,y,xy) & mult(y,z,yz) & mult(xy,z,xyz) & mult(x,yz,xyz2)) -> xyz = xyz2)",
            "exists e (all x (mult(e,x,x) & mult(x,e,x)))",
        ]

    @staticmethod
    def group_axioms() -> List[str]:
        """Axioms for a group"""
        return CategoricalHelpers.monoid_axioms() + [
            "all x exists y (mult(x,y,e) & mult(y,x,e))"
        ]

class LogicCore:
    """
    Core logic engine integrating Prover9 and Mace4 for formal verification.
    Ported from mcp-logic to run directly within RAA.
    """
    def __init__(self, prover_path: Optional[str] = None):
        """Initialize connection to Prover9 and Mace4"""
        if prover_path is None:
            # Default to internal bin directory: src/cognition/ladr/bin
            root_dir = os.path.dirname(os.path.abspath(__file__))
            prover_path = os.path.join(root_dir, "ladr", "bin")

            # Check if environment variable override exists
            env_path = os.getenv("PROVER9_PATH")
            if env_path:
                prover_path = env_path

        self.prover_path = Path(prover_path)

        # Initialize Prover9
        self.prover_exe = self.prover_path / "prover9.exe"
        if not self.prover_exe.exists():
            self.prover_exe = self.prover_path / "prover9"

        if not self.prover_exe.exists():
            logger.warning(f"Prover9 not found at {self.prover_exe}. Logical verification will be disabled.")
            self.prover_exe = None
            self.mace4_exe = None
            return

        logger.info(f"Initialized LogicCore with Prover9 at {self.prover_exe}")

        # Initialize Mace4
        self.mace4_exe = self.prover_path / "mace4.exe"
        if not self.mace4_exe.exists():
            self.mace4_exe = self.prover_path / "mace4"
            if not self.mace4_exe.exists():
                logger.warning(f"Mace4 not found at {self.mace4_exe}. Counterexample search disabled.")
                self.mace4_exe = None
            else:
                logger.debug(f"Mace4 initialized at {self.mace4_exe}")

        self.validator = SyntaxValidator()
        self.cat_helpers = CategoricalHelpers()

    def check_well_formed(self, formulas: List[str]) -> Dict[str, Any]:
        """Validate syntax of formulas"""
        results = {"valid": True, "formula_results": []}

        for i, formula in enumerate(formulas):
            is_valid, errors, warnings = self.validator.validate(formula)
            formula_result = {"formula": formula, "valid": is_valid, "errors": errors, "warnings": warnings}
            results["formula_results"].append(formula_result)
            if not is_valid:
                results["valid"] = False

        return results

    def verify_commutativity(self, path_a: List[str], path_b: List[str],
                            object_start: str, object_end: str,
                            with_category_axioms: bool = True) -> Dict[str, Any]:
        """Verify diagram commutativity"""
        premises, conclusion = self.cat_helpers.verify_commutativity(path_a, path_b, object_start, object_end)

        if with_category_axioms:
            cat_axioms = self.cat_helpers.category_axioms()
            premises = cat_axioms + premises

        return {"premises": premises, "conclusion": conclusion}

    def get_category_axioms(self, concept: str, **kwargs) -> List[str]:
        """Get axioms for a categorical concept"""
        if concept == "category":
            return self.cat_helpers.category_axioms()
        elif concept == "functor":
            return self.cat_helpers.functor_axioms(kwargs.get("functor_name", "F"))
        elif concept == "natural-transformation":
            return self.cat_helpers.natural_transformation_condition(
                kwargs.get("functor_f", "F"),
                kwargs.get("functor_g", "G"),
                kwargs.get("component", "alpha")
            )
        elif concept == "monoid":
            return self.cat_helpers.monoid_axioms()
        elif concept == "group":
            return self.cat_helpers.group_axioms()
        return []

    def prove(self, premises: List[str], conclusion: str, timeout: int = 30) -> Dict[str, Any]:
        """Prove a logical statement using Prover9."""
        if not self.prover_exe:
            return {"result": "error", "reason": "Prover9 not available"}

        try:
            # Validate first
            valid_check = self.check_well_formed(premises + [conclusion])
            if not valid_check["valid"]:
                 return {"result": "error", "reason": "Syntax error in formulas", "details": valid_check}

            input_file = self._create_prover_input_file(premises, conclusion)
            return self._run_prover(input_file, timeout)
        except Exception as e:
            logger.error(f"Proof error: {e}")
            return {"result": "error", "reason": str(e)}

    def find_counterexample(self, premises: List[str], conclusion: str, domain_size: Optional[int] = None) -> Dict[str, Any]:
        """Find a counterexample showing the conclusion doesn't follow from premises using Mace4."""
        if not self.mace4_exe:
            return {"result": "error", "reason": "Mace4 not available"}

        try:
            # Validate first
            valid_check = self.check_well_formed(premises + [conclusion])
            if not valid_check["valid"]:
                 return {"result": "error", "reason": "Syntax error in formulas", "details": valid_check}

            input_file = self._create_mace_input_file(premises, goal=conclusion, domain_size=domain_size)
            result = self._run_mace4(input_file)

            if result["result"] == "model_found":
                result["interpretation"] = f"Counterexample found: The premises are satisfied but the conclusion '{conclusion}' is FALSE in this model."

            return result
        except Exception as e:
            logger.error(f"Counterexample search error: {e}")
            return {"result": "error", "reason": str(e)}

    def find_model(self, premises: List[str], domain_size: Optional[int] = None) -> Dict[str, Any]:
        """Find a finite model satisfying the premises"""
        if not self.mace4_exe:
            return {"result": "error", "reason": "Mace4 not available"}

        try:
            # Validate
            valid_check = self.check_well_formed(premises)
            if not valid_check["valid"]:
                 return {"result": "error", "reason": "Syntax error in formulas", "details": valid_check}

            input_file = self._create_mace_input_file(premises, goal=None, domain_size=domain_size)
            return self._run_mace4(input_file)
        except Exception as e:
            logger.error(f"Model search error: {e}")
            return {"result": "error", "reason": str(e)}

    def _create_prover_input_file(self, premises: List[str], goal: str) -> Path:
        """Create a Prover9 input file"""
        content = ["formulas(assumptions)."]
        content.extend([p if p.endswith(".") else p + "." for p in premises])
        content.append("end_of_list.")
        content.append("")
        content.append("formulas(goals).")
        content.append(goal if goal.endswith(".") else goal + ".")
        content.append("end_of_list.")

        input_content = "\n".join(content)
        fd, path = tempfile.mkstemp(suffix=".in", text=True)
        with os.fdopen(fd, "w") as f:
            f.write(input_content)
        return Path(path)

    def _create_mace_input_file(self, premises: List[str], goal: Optional[str] = None, domain_size: Optional[int] = None) -> Path:
        """Create a Mace4 input file"""
        content = []

        # Domain size configuration
        if domain_size is not None:
            content.append(f"assign(domain_size, {domain_size}).")
        else:
            # Incremental search defaults often handled by mace4 binary flags (-N),
            # but defining start/end in file is safer for wrapper
            content.append("assign(start_size, 2).")
            content.append("assign(end_size, 10).")

        content.append("formulas(assumptions).")
        content.extend([p if p.endswith(".") else p + "." for p in premises])
        content.append("end_of_list.")

        if goal:
            content.append("")
            content.append("formulas(goals).")
            content.append(goal if goal.endswith(".") else goal + ".")
            content.append("end_of_list.")

        input_content = "\n".join(content)
        fd, path = tempfile.mkstemp(suffix=".in", text=True)
        with os.fdopen(fd, "w") as f:
            f.write(input_content)
        return Path(path)

    def _run_prover(self, input_path: Path, timeout: int = 60) -> Dict[str, Any]:
        """Run Prover9 executable"""
        try:
            # Set working directory to Prover9 directory for includes if needed
            cwd = str(self.prover_path)
            result = subprocess.run(
                [str(self.prover_exe), "-f", str(input_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )

            if "THEOREM PROVED" in result.stdout:
                # Extract proof
                try:
                    proof = result.stdout.split("PROOF =")[1].split("====")[0].strip()
                except IndexError:
                    proof = "Theorem proved but proof extraction failed."
                return {"result": "proved", "proof": proof}
            elif "SEARCH FAILED" in result.stdout:
                return {"result": "unprovable", "reason": "Proof search failed", "output": result.stdout[:500]}
            elif "Fatal error" in result.stderr:
                 return {"result": "error", "reason": "Syntax or Runtime error", "error": result.stderr}
            else:
                 return {"result": "uncertain", "output": result.stdout[:500]}

        except subprocess.TimeoutExpired:
            return {"result": "timeout", "reason": f"Exceeded {timeout}s"}
        except Exception as e:
             return {"result": "error", "reason": str(e)}
        finally:
            try:
                input_path.unlink()
            except OSError:
                pass

    def _run_mace4(self, input_path: Path, timeout: int = 30) -> Dict[str, Any]:
        """Run Mace4 executable"""
        try:
            cwd = str(self.prover_path)
            # -c means compile/print models cleanly
            result = subprocess.run(
                [str(self.mace4_exe), "-f", str(input_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )

            output = result.stdout
            if "Process 1 exit (max_models)" in output or "Model 1" in output:
                # Extract the model
                # Simplistic extraction:
                return {"result": "model_found", "raw_output": output}
            elif "Process 1 exit (max_seconds)" in output:
                return {"result": "timeout"}
            else:
                return {"result": "no_model_found"}

        except subprocess.TimeoutExpired:
            return {"result": "timeout"}
        except Exception as e:
            return {"result": "error", "reason": str(e)}
        finally:
            try:
                input_path.unlink()
            except OSError:
                pass
