import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;
import java.util.Map;


class NoOfSubsequences {
    /**
     * * Write a function to find the number of subsequences having product smaller than var_tfgp for the given non negative array.
     *
     * > var_58dw([1, 2, 3, 4], 10)
     * 11
     * > var_58dw([4, 8, 7, 2], 50)
     * 9
     * > var_58dw([5, 6, 7, 8], 15)
     * 4
     */
    public static int var_58dw(List<Integer> var_kjfp, int var_tfgp) {

        // Write your code here.
        int var_gbnq = var_kjfp.size();
        int[][] dp = new int[var_tfgp + 1][var_gbnq + 1];

        for (int var_hqzo = 1; var_hqzo <= var_tfgp; var_hqzo++) {

            for (int var_hczt = 1; var_hczt <= var_gbnq; var_hczt++) {
                dp[var_hqzo][var_hczt] = dp[var_hqzo][var_hczt - 1];
                if (var_kjfp.get(var_hczt - 1) <= var_hqzo && var_kjfp.get(var_hczt - 1) > 0) {

                    dp[var_hqzo][var_hczt] += dp[var_hqzo / var_kjfp.get(var_hczt - 1)][var_hczt - 1] + 1;
                }
            }
        }


        return dp[var_tfgp][var_gbnq];
    }
}