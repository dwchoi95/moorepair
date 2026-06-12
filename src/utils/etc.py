import ast


class _TreeNode:
    """Lightweight AST node wrapper for APTED tree edit distance."""
    __slots__ = ("name", "children")

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
        # Salient identifiers/values so renames count as edits
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
        children = [ETC._ast_to_tree(c) for c in ast.iter_child_nodes(node)]
        return _TreeNode(label, children)

    @staticmethod
    def _tree_size(node: _TreeNode) -> int:
        return 1 + sum(ETC._tree_size(c) for c in node.children)

    @staticmethod
    def rps(buggy_code: str, patch_code: str) -> float:
        """Relative Patch Size: AST tree edit distance between the buggy
        program and the patch, normalized by the AST size of the buggy
        program (Refactory-style). Falls back to a normalized string
        Levenshtein distance when either program fails to parse."""
        try:
            t_buggy = ETC._ast_to_tree(ast.parse(buggy_code))
            t_patch = ETC._ast_to_tree(ast.parse(patch_code))
        except (SyntaxError, ValueError, RecursionError):
            import Levenshtein
            a = ETC.normalize_code(buggy_code)
            b = ETC.normalize_code(patch_code)
            return ETC.divide(Levenshtein.distance(a, b), len(a))

        from apted import APTED, Config

        class _AptedConfig(Config):
            def rename(self, n1, n2):
                return 0 if n1.name == n2.name else 1

            def children(self, n):
                return n.children

        ted = APTED(t_buggy, t_patch, _AptedConfig()).compute_edit_distance()
        return ETC.divide(ted, ETC._tree_size(t_buggy))
