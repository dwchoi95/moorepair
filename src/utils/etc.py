import ast
from apted import APTED
import Levenshtein

class _TreeNode:
    def __init__(self, name: str, children: list):
        self.name = name
        self.children = children


class ETC:
    @staticmethod
    def calc_lcs(lst_a, lst_b) -> int:
        # Calculate LCS
        m, n = len(lst_a), len(lst_b)
        dp = [[0 for j in range(n+1)] for i in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if lst_a[i-1] == lst_b[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    @staticmethod
    def divide(a, b):
        try: res = a/b
        except ZeroDivisionError:
            res = 0
        return res
    
    @staticmethod
    def normalize_lines(code: str) -> list:
        return ["".join(line.split()) for line in code.splitlines() if line.strip()]
    
    @staticmethod
    def normalize_code(code: str) -> str:
        # One-line normalization to ignore formatting-only differences
        # (spaces, tabs, newlines) across generated variants.
        return "".join(ETC.normalize_lines(code))

    # ---------------------------------------------------------------- #
    # RPS — Relative Patch Size                                        #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _ast_to_tree(node: ast.AST) -> _TreeNode:
        label = type(node).__name__
        children = [ETC._ast_to_tree(c) for c in ast.iter_child_nodes(node)]
        return _TreeNode(label, children)

    @staticmethod
    def _tree_size(node: _TreeNode) -> int:
        return 1 + sum(ETC._tree_size(c) for c in node.children)

    @staticmethod
    def rps(buggy_code: str, patch_code: str) -> float:
        """Relative Patch Size (RPS) = TED / |AST(buggy)|
        where TED = tree edit distance between AST(buggy) and AST(patch).
        """
        t_buggy = ETC._ast_to_tree(ast.parse(buggy_code))
        t_patch = ETC._ast_to_tree(ast.parse(patch_code))

        ted = APTED(t_buggy, t_patch).compute_edit_distance()
        return ETC.divide(ted, ETC._tree_size(t_buggy))
    
    @staticmethod
    def _ast_label(node: "ast.AST") -> str:
        label = type(node).__name__
        if isinstance(node, ast.Name):
            label += f":{node.id}"
        elif isinstance(node, ast.Constant):
            label += f":{node.value!r}"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            label += f":{node.name}"
        elif isinstance(node, ast.arg):
            label += f":{node.arg}"
        elif isinstance(node, ast.Attribute):
            label += f":{node.attr}"
        return label

    @staticmethod
    def _ast_labels(code: str):
        """DFS (preorder) sequence of AST node labels; None if unparseable."""
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError):
            return None
        seq = []
        stack = [tree]
        while stack:
            node = stack.pop()
            seq.append(ETC._ast_label(node))
            children = list(ast.iter_child_nodes(node))
            stack.extend(reversed(children))  # preorder
        return seq

    @staticmethod
    def edit_ratio(code_a: str, code_b: str) -> float:
        """Fast edit-distance ratio between two programs: normalized character
        Levenshtein on whitespace-stripped code, divided by the length of
        code_a. Stand-in used when either side won't parse."""
        a = ETC.normalize_code(code_a)
        b = ETC.normalize_code(code_b)
        return ETC.divide(Levenshtein.distance(a, b), max(len(a), 1))

    @staticmethod
    def ted(buggy_code: str, patch_code: str) -> float:
        """f_sim: structural similarity to the buggy program. Levenshtein edit
        distance over the DFS (preorder)-linearized AST node-label sequences,
        normalized by the length of the buggy program's sequence, so smaller
        means structurally closer to the buggy. Ignores whitespace/formatting;
        falls back to character edit_ratio when either side won't parse."""
        la = ETC._ast_labels(buggy_code)
        lb = ETC._ast_labels(patch_code)
        if la is None or lb is None:
            return ETC.edit_ratio(buggy_code, patch_code)
        # Map each distinct label to a unique char, then run C-level
        # Levenshtein on the encoded strings (= sequence edit distance).
        vocab: dict = {}
        def enc(seq):
            out = []
            for lab in seq:
                if lab not in vocab:
                    vocab[lab] = chr(0x100 + len(vocab))
                out.append(vocab[lab])
            return "".join(out)
        sa, sb = enc(la), enc(lb)
        # Paper f_sim: normalize by max(|sigma_b|, |sigma_p|) so it lies in [0,1].
        return ETC.divide(Levenshtein.distance(sa, sb), max(len(sa), len(sb), 1))