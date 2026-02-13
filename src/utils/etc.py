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
