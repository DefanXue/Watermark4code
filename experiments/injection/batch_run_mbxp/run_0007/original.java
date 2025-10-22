import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class NoOfSubsequences {
    /**
     * * Write a function to find the number of subsequences having product smaller than k for the given non negative array.
     *
     * > noOfSubsequences([1, 2, 3, 4], 10)
     * 11
     * > noOfSubsequences([4, 8, 7, 2], 50)
     * 9
     * > noOfSubsequences([5, 6, 7, 8], 15)
     * 4
     */
    public static int noOfSubsequences(List<Integer> arr, int k) {
        // Write your code here.
        int n = arr.size();
        int[][] dp = new int[k + 1][n + 1];

        for (int i = 1; i <= k; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = dp[i][j - 1];
                if (arr.get(j - 1) <= i && arr.get(j - 1) > 0) {
                    dp[i][j] += dp[i / arr.get(j - 1)][j - 1] + 1;
                }
            }
        }

        return dp[k][n];
    }
}