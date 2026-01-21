import ast
from apted.apted import APTED
from apted.helpers import Tree
from functools import cache

class TED:
    @classmethod
    def clear_cache(cls):
        cls._to_tree.cache_clear()
        cls.compute_ted.cache_clear()
        cls.compute_sim.cache_clear()
        cls.compute_ast_size.cache_clear()
        cls.relative_patch_size.cache_clear()
        
    @classmethod
    def __compute_ast_size(cls, tree):
        """
        Computes the size of the AST tree.
        """
        return 1 + sum(cls.__compute_ast_size(child) for child in tree.children)
    
    @classmethod
    def __ast_to_apted(cls, node):
        node_label = type(node).__name__
        children = [cls.__ast_to_apted(child) for child in ast.iter_child_nodes(node)]
        if children:
            return "{" + node_label + "".join(children) + "}"
        else:
            return "{" + node_label + "}"
    
    @classmethod
    @cache
    def _to_tree(cls, code):
        tree = ast.parse(code)
        tree_str = cls.__ast_to_apted(tree)
        return Tree.from_text(tree_str)
        
    @classmethod
    @cache
    def compute_ted(cls, code1, code2):
        """
        Computes the tree edit distance between two pieces of Python code with APTED.
        """
        tree1 = cls._to_tree(code1)
        tree2 = cls._to_tree(code2)
        apted = APTED(tree1, tree2)
        return apted.compute_edit_distance()
    
    @classmethod
    @cache
    def compute_sim(cls, code1, code2):
        """
        Computes the similarity between two pieces of Python code based on tree edit distance with APTED.
        """
        tree1 = cls._to_tree(code1)
        tree2 = cls._to_tree(code2)
        apted = APTED(tree1, tree2)
        distance = apted.compute_edit_distance()
        max_size = max(cls.__compute_ast_size(tree1), cls.__compute_ast_size(tree2))
        similarity = 1 - (distance / max_size) if max_size > 0 else 1.0
        return similarity
    
    @classmethod
    @cache
    def compute_ast_size(cls, code):
        tree = cls._to_tree(code)
        return cls.__compute_ast_size(tree)
    
    @classmethod
    @cache
    def relative_patch_size(cls, buggy, patch):
        buggy_tree = cls._to_tree(buggy)
        patch_tree = cls._to_tree(patch)
        apted = APTED(buggy_tree, patch_tree)
        ted = apted.compute_edit_distance()
        buggy_size = cls.__compute_ast_size(buggy_tree)
        return round(ted / buggy_size, 2)


# Example usage
class TEDTest:
    @staticmethod
    def run(code1, code2):
        distance = TED.compute_ted(code1, code2)
        print(f"Distance: {distance}")
        similarity = TED.compute_sim(code1, code2)
        print(f"Similarity: {similarity}")
        rps = TED.relative_patch_size(code1, code2)
        print(f"RPS: {rps}")
