import ast
import re
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from codebleu import calc_codebleu
from src.genetic import Fitness, Variation
from src.execution import Programs, Program, Tester
from src.utils import ETC


class PaREffiLearner:
    def __init__(
        self,
        buggys: Programs,
        references: Programs,
        assignement: dict,
    ):
        self.buggys = buggys
        self.references = references
        self.assignement = assignement
        self.variation = Variation(assignement)
        self._patch_uid = 0

        self.bm25 = BM25Okapi([
            self._anonymize_code(ref.code).split() 
            for ref in self.references])
    
    def _assign_patch_id(self, patch: Program) -> None:
        self._patch_uid += 1
        patch.id = f"pop_{self._patch_uid}"
        
    def _syntax_check(self, program: Program) -> bool:
        try:
            ast.parse(program.code)
            return True
        except Exception: pass
        return False
    
    def _match_tc(self, buggy: Program, reference: Program) -> float:
        buggy_results = Tester.run(buggy)
        ref_results = Tester.run(reference)
        buggy_passed, _ = Tester.tests_split(buggy_results)
        ref_passed, _ = Tester.tests_split(ref_results)
        clip = len(buggy_passed & ref_passed)
        denom = len(ref_passed) + len(buggy_passed)
        return ETC.divide(2 * clip, denom)

    def _match_codebleu(self, buggy: str, reference: str) -> tuple[float, float]:
        scores = calc_codebleu(
            references=[reference],
            predictions=[buggy],
            lang="python",
        )
        return scores['dataflow_match_score'], scores['syntax_match_score']

    @staticmethod
    def _anonymize_code(code: str) -> str:
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
        Fitness.evaluate(buggy)
        
        reference = self._get_reference(buggy)
        for gen in tqdm(range(1, generations + 1), desc="Generation", position=1, leave=False):
            result.setdefault(gen, solutions.copy())
            patch = self.variation.correct(buggy, [reference])
            if not patch: continue
            patch = patch[0]
            passed = False
            if self._syntax_check(patch):
                results = Tester.run(patch, profiling=True)
                passed = Tester.is_all_pass(results)
            if not passed: continue
            valids = [patch] * pop_size
            efficients = self.variation.efficient(valids)
            for patch in efficients:
                results = Tester.run(patch)
                passed = Tester.is_all_pass(results)
                if passed: 
                    self._assign_patch_id(patch)
                    solutions.append(patch)
        return result
                    
    def run(self, generations: int = 5, pop_size: int = 6) -> dict:
        results = {}
        for buggy in tqdm(self.buggys, desc="Buggy", position=0):
            results[buggy.id] = self._run_single(buggy, generations, pop_size)
        return results
