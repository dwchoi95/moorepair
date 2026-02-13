import tree_sitter_c as tsc
from tree_sitter import Language, Parser
from apted.apted import APTED
from apted.helpers import Tree
from functools import cache
import Levenshtein
import logging
from codebleu import calc_codebleu

from .etc import ETC


class Distance:
    LANGUAGE = None
    LANGUAGE_PACKAGES = {
        'c': tsc.language(),
    }

    @classmethod
    def set_language(cls, language:str):
        language = language.lower()
        if cls.LANGUAGE == language: return
        cls.LANGUAGE = language
        if cls.LANGUAGE not in cls.LANGUAGE_PACKAGES:
            raise ValueError(f"Unsupported language: {cls.LANGUAGE}")
        lang = Language(cls.LANGUAGE_PACKAGES[cls.LANGUAGE])
        cls.parser = Parser(lang)

    @classmethod
    def clear_cache(cls):
        cls._to_tree.cache_clear()
        cls._to_node_sequence.cache_clear()
        cls.codebleu.cache_clear()

    @classmethod
    def __compute_ast_size(cls, tree):
        """
        Computes the size of the AST tree.
        """
        return 1 + sum(cls.__compute_ast_size(child) for child in tree.children)

    @classmethod
    def __node_to_apted(cls, node):
        node_label = node.type
        children = []
        for child in node.children:
            if child.is_named:
                children.append(cls.__node_to_apted(child))
        if children:
            return "{" + node_label + "".join(children) + "}"
        else:
            return "{" + node_label + "}"

    @classmethod
    @cache
    def _to_tree(cls, code):
        tree = cls.parser.parse(bytes(code, "utf-8"))
        tree_str = cls.__node_to_apted(tree.root_node)
        return Tree.from_text(tree_str)
    
    @classmethod
    def __node_to_sequence(cls, node) -> list:
        """
        Converts AST to a sequence of node types (pre-order traversal).
        """
        nodes = []
        nodes.append(node.type)
        for child in node.children:
            if child.is_named:
                nodes.extend(cls.__node_to_sequence(child))
        return nodes

    @classmethod
    @cache
    def _to_node_sequence(cls, code):
        """
        Converts code to a sequence of AST node types.
        """
        tree = cls.parser.parse(bytes(code, "utf-8"))
        return cls.__node_to_sequence(tree.root_node)

    @classmethod
    def compute_ted(cls, code1, code2):
        """
        Computes the tree edit distance between two pieces of code with APTED.
        """
        tree1 = cls._to_tree(code1)
        tree2 = cls._to_tree(code2)
        apted = APTED(tree1, tree2)
        return apted.compute_edit_distance()

    @classmethod
    def compute_ccd(cls, coverage1:dict, coverage2:dict):
        """
        Computes the distance between two code coverage using Jaccard Similarity.
        """
        ccd = 0
        for tc_id, cov1 in coverage1.items():
            cov2 = coverage2[tc_id]
            intersection = len(cov1 & cov2)
            union = len(cov1 | cov2)
            jaccard_sim = intersection / union if union != 0 else 1.0
            ccd += 1.0 - jaccard_sim
        return ccd / len(coverage1) if ccd > 0 else 0.0
    
    @classmethod
    def compute_levenshtein_ted(cls, code1, code2):
        """
        Computes the edit distance between two ASTs using Levenshtein distance
        on the sequence of node types.
        """
        seq1 = cls._to_node_sequence(code1)
        seq2 = cls._to_node_sequence(code2)
        seq_set = set(seq1).union(set(seq2))
        char_map = {node: chr(i + 1) for i, node in enumerate(seq_set)}
        str1 = ''.join(char_map[node] for node in seq1)
        str2 = ''.join(char_map[node] for node in seq2)
        return Levenshtein.distance(str1, str2)

    @classmethod
    def compute_sim(cls, code1, code2):
        """
        Computes the similarity between two pieces of code based on tree edit distance with APTED.
        """
        tree1 = cls._to_tree(code1)
        tree2 = cls._to_tree(code2)
        apted = APTED(tree1, tree2)
        distance = apted.compute_edit_distance()
        max_size = max(cls.__compute_ast_size(tree1), cls.__compute_ast_size(tree2))
        similarity = 1 - (distance / max_size) if max_size > 0 else 1.0
        return similarity

    @classmethod
    def compute_ast_size(cls, code):
        tree = cls._to_tree(code)
        return cls.__compute_ast_size(tree)

    @classmethod
    def relative_patch_size(cls, buggy, patch):
        buggy_tree = cls._to_tree(buggy)
        patch_tree = cls._to_tree(patch)
        apted = APTED(buggy_tree, patch_tree)
        ted = apted.compute_edit_distance()
        buggy_size = cls.__compute_ast_size(buggy_tree)
        return round(ted / buggy_size, 2)
    
    @classmethod
    @cache
    def codebleu(cls, code1, code2):
        if code2 is None or code2.strip() == "":
            return 0.0
        logging.disable(logging.WARNING)
        try:
            return calc_codebleu(
                [code1],
                [code2],
                lang=cls.LANGUAGE
            )["codebleu"]
        finally:
            logging.disable(logging.NOTSET)



# Example usage
class TEDTest:
    @staticmethod
    def run(code1, code2):
        TED.set_language(language='c')
        distance = TED.compute_ted(code1, code2)
        print(f"TED Distance: {distance}")
        similarity = TED.compute_sim(code1, code2)
        print(f"TED Similarity: {similarity}")
        rps = TED.relative_patch_size(code1, code2)
        print(f"TED RPS: {rps}")
        lcs = TED.compute_levenshtein_ted(code1, code2)
        print(f"Levenshtein-based TED Distance: {lcs}")
