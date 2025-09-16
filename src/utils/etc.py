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
