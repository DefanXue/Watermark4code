import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;
import java.util.Map;
import java.util.List;
import java.util.Set;
import java.util.List;


class NoOfSubsequences {


    /**
     * * Write a function to find the number of subsequences having product smaller than var_1luw for the given non negative array.
     *
     * > var_vga2([1, 2, 3, 4], 10)
     * 11
     * > var_vga2([4, 8, 7, 2], 50)
     * 9
     * > var_vga2([5, 6, 7, 8], 15)
     * 4
     */
    public static int var_vga2(List<Integer> var_urml, int var_1luw) {








        // Write your code here.
        int var_fxud = var_urml.size();
        int[][] dp = new int[((((var_1luw + 1))))][((((var_fxud + 1))))];

        for (int var_gqyc = 1; var_gqyc <= var_1luw; var_gqyc++) {




            for (int var_tq8s = 1; var_tq8s <= var_fxud; var_tq8s++) {






                dp[var_gqyc][var_tq8s] = dp[var_gqyc][var_tq8s - 1];
                if (var_urml.get(((var_tq8s - 1))) <= var_gqyc && var_urml.get((var_tq8s - 1)) > 0) {







                    dp[var_gqyc][var_tq8s] += dp[(((((var_gqyc / var_urml))))).get((((var_tq8s - 1))))][((var_tq8s - 1))] + 1;
                }
            }
        }





        return dp[var_1luw][var_fxud];
    }
}