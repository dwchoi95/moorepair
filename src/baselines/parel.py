import ast
import re
import logging
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from codebleu import calc_codebleu
from src.genetic import Fitness, Variation
from src.execution import Programs, Program, Tester, Status
from src.utils import ETC


class PaREffiLearner:
    def __init__(
        self,
        buggys: Programs,
        references: Programs,
        description: str,
        log_path: str = "logs/temp.log",
    ):
        self.buggys = buggys
        self.references = references
        self.description = description
        self.variation = Variation(description)
        self._patch_uid = 0

        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("PaREffiLearner")
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
            h.close()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(fh)
        self.logger.propagate = False
        
        self.bm25 = BM25Okapi([
            self._anonymize_code(ref.code).split() 
            for ref in self.references])
    
    def _match_tc(self, buggy: Program, reference: Program) -> float:
        """Test Cases Pass Match Score: 2 * clip / (count_correct + count_buggy)"""
        buggy_results = Tester.run(buggy)
        ref_results = Tester.run(reference)
        buggy_passed, _ = Tester.tests_split(buggy_results)
        ref_passed, _ = Tester.tests_split(ref_results)
        clip = len(buggy_passed & ref_passed)
        denom = len(ref_passed) + len(buggy_passed)
        return ETC.divide(2 * clip, denom)

    def _match_codebleu(self, buggy: str, reference: str) -> tuple[float, float]:
        """Data-flow Match Score via CodeBLEU's dataflow_match."""
        """AST Match Score via CodeBLEU's syntax_match."""
        scores = calc_codebleu(
            references=[reference],
            predictions=[buggy],
            lang="python",
        )
        return scores['dataflow_match_score'], scores['syntax_match_score']

    @staticmethod
    def _anonymize_code(code: str) -> str:
        """Replace variable names with generic names (v1, v2, ...) for BM25."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                names.add(node.name)
                for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                    names.add(arg.arg)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
        # Sort by length descending to avoid partial replacements
        mapping = {}
        for i, name in enumerate(sorted(names, key=len, reverse=True)):
            mapping[name] = f"v{i}"
        result = code
        for orig, anon in mapping.items():
            result = re.sub(r'\b' + re.escape(orig) + r'\b', anon, result)
        return result

    def _bm25_anon(self, buggy: Program, reference: Program) -> float:
        """Anonymous BM25 Score, normalized across all references."""
        anon_buggy = self._anonymize_code(buggy.code).split()
        scores = self.bm25.get_scores(anon_buggy)
        bm25_min, bm25_max = float(scores.min()), float(scores.max())
        if bm25_max == bm25_min:
            return 0.0
        # Find the index of this reference in self.references
        ref_idx = None
        for i, ref in enumerate(self.references):
            if ref.id == reference.id:
                ref_idx = i
                break
        if ref_idx is None:
            return 0.0
        return (float(scores[ref_idx]) - bm25_min) / (bm25_max - bm25_min)

    def _get_reference(self, buggy: Program) -> Program:
        """Select the peer solution with the highest PSM score."""
        best_refer = None
        best_psm = -1.0
        for refer in self.references:
            a = self._match_tc(buggy, refer)
            b, c = self._match_codebleu(buggy.code, refer.code)
            d = self._bm25_anon(buggy, refer)
            psm = 0.25 * (a + b + c + d)
            if psm > best_psm:
                best_psm = psm
                best_refer = refer
        return best_refer
    
    def _run_single(self, buggy: Program, generations: int, pop_size: int) -> dict:
        result = {}
        solutions = []
        buggy_fitness = Fitness.evaluate(buggy)
        
        self.logger.info(f"Buggy: {buggy.id}\n{buggy.code}\n")
        reference = self._get_reference(buggy)
        for gen in tqdm(range(1, generations + 1), desc="Generation", position=1, leave=False):
            result.setdefault(gen, solutions.copy())
            self.logger.info(f"=== Generation {gen} ===")
            valids = []
            corrects = self.variation.correct(buggy, [reference], pop_size)
            for patch in corrects:
                results = Tester.run(patch)
                passed = Tester.is_all_pass(results)
                self.logger.info(
                    f"PaR: {Status.PASSED if passed else Status.FAILED}\n{patch.code}\n")
                if passed:
                    results = Tester.run(patch, profile=True)
                    valids.append(patch)
            efficients = self.variation.efficient(valids)
            for patch in efficients:
                results = Tester.run(patch)
                passed = Tester.is_all_pass(results)
                self.logger.info(
                    f"EffiLearner: {Status.PASSED if passed else Status.FAILED}\n{patch.code}\n")
                if passed:
                    solutions.append(patch)
        return result
                    
    def run(self, generations: int = 5, pop_size: int = 5) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._run_single(buggy, generations, pop_size)
        return results
