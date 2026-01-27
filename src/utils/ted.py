import tree_sitter_c as tsc
from tree_sitter import Language, Parser
from apted.apted import APTED
from apted.helpers import Tree
from functools import cache
import Levenshtein
import logging
from codebleu import calc_codebleu

class TED:
    LANGUAGE_PACKAGES = {
        'c': tsc.language(),
    }

    def __init__(self, language: str):
        self.language = language.lower()
        if self.language not in self.LANGUAGE_PACKAGES:
            raise ValueError(f"Unsupported language: {self.language}")
        lang = Language(self.LANGUAGE_PACKAGES[self.language])
        self.parser = Parser(lang)

    def clear_cache(self):
        self._to_tree.cache_clear()
        self.compute_ted.cache_clear()
        self.compute_sim.cache_clear()
        self.compute_ast_size.cache_clear()
        self.relative_patch_size.cache_clear()
        self._to_node_sequence.cache_clear()
        self.compute_levenshtein_ted.cache_clear()
        self.compute_levenshtein_led.cache_clear()

    def __compute_ast_size(self, tree):
        """
        Computes the size of the AST tree.
        """
        return 1 + sum(self.__compute_ast_size(child) for child in tree.children)

    def __node_to_apted(self, node):
        node_label = node.type
        children = []
        for child in node.children:
            if child.is_named:
                children.append(self.__node_to_apted(child))
        if children:
            return "{" + node_label + "".join(children) + "}"
        else:
            return "{" + node_label + "}"

    @cache
    def _to_tree(self, code):
        tree = self.parser.parse(bytes(code, "utf-8"))
        tree_str = self.__node_to_apted(tree.root_node)
        return Tree.from_text(tree_str)
    
    def __node_to_sequence(self, node) -> list:
        """
        Converts AST to a sequence of node types (pre-order traversal).
        """
        nodes = []
        nodes.append(node.type)
        for child in node.children:
            if child.is_named:
                nodes.extend(self.__node_to_sequence(child))
        return nodes

    @cache
    def _to_node_sequence(self, code):
        """
        Converts code to a sequence of AST node types.
        """
        tree = self.parser.parse(bytes(code, "utf-8"))
        return self.__node_to_sequence(tree.root_node)

    @cache
    def compute_ted(self, code1, code2):
        """
        Computes the tree edit distance between two pieces of code with APTED.
        """
        tree1 = self._to_tree(code1)
        tree2 = self._to_tree(code2)
        apted = APTED(tree1, tree2)
        return apted.compute_edit_distance()

    @cache
    def compute_levenshtein_led(self, code1:str, code2:str):
        """
        Computes the edit distance between two Line(s) using Levenshtein distance
        on the sequence of node types.
        """
        seq1 = [line for line in code1.splitlines() if line.strip()]
        seq2 = [line for line in code2.splitlines() if line.strip()]
        seq_set = set(seq1).union(set(seq2))
        char_map = {node: chr(i + 1) for i, node in enumerate(seq_set)}
        str1 = ''.join(char_map[node] for node in seq1)
        str2 = ''.join(char_map[node] for node in seq2)
        return Levenshtein.distance(str1, str2)
    
    @cache
    def compute_levenshtein_ted(self, code1, code2):
        """
        Computes the edit distance between two ASTs using Levenshtein distance
        on the sequence of node types.
        """
        seq1 = self._to_node_sequence(code1)
        seq2 = self._to_node_sequence(code2)
        seq_set = set(seq1).union(set(seq2))
        char_map = {node: chr(i + 1) for i, node in enumerate(seq_set)}
        str1 = ''.join(char_map[node] for node in seq1)
        str2 = ''.join(char_map[node] for node in seq2)
        return Levenshtein.distance(str1, str2)

    @cache
    def compute_sim(self, code1, code2):
        """
        Computes the similarity between two pieces of code based on tree edit distance with APTED.
        """
        tree1 = self._to_tree(code1)
        tree2 = self._to_tree(code2)
        apted = APTED(tree1, tree2)
        distance = apted.compute_edit_distance()
        max_size = max(self.__compute_ast_size(tree1), self.__compute_ast_size(tree2))
        similarity = 1 - (distance / max_size) if max_size > 0 else 1.0
        return similarity

    @cache
    def compute_ast_size(self, code):
        tree = self._to_tree(code)
        return self.__compute_ast_size(tree)

    @cache
    def relative_patch_size(self, buggy, patch):
        buggy_tree = self._to_tree(buggy)
        patch_tree = self._to_tree(patch)
        apted = APTED(buggy_tree, patch_tree)
        ted = apted.compute_edit_distance()
        buggy_size = self.__compute_ast_size(buggy_tree)
        return round(ted / buggy_size, 2)
    
    @cache
    def codebleu(self, code1, code2):
        if code2 is None or code2.strip() == "":
            return 0.0
        logging.disable(logging.WARNING)
        try:
            return calc_codebleu(
                [code1],
                [code2],
                lang=self.language
            )["codebleu"]
        finally:
            logging.disable(logging.NOTSET)



# Example usage
class TEDTest:
    @staticmethod
    def run(code1, code2):
        ted = TED(language='c')
        distance = ted.compute_ted(code1, code2)
        print(f"TED Distance: {distance}")
        similarity = ted.compute_sim(code1, code2)
        print(f"TED Similarity: {similarity}")
        rps = ted.relative_patch_size(code1, code2)
        print(f"TED RPS: {rps}")
        lcs = ted.compute_levenshtein_ted(code1, code2)
        print(f"Levenshtein-based TED Distance: {lcs}")
