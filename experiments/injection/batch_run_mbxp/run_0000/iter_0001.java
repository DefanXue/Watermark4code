import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class MagicSquareTest {
    /**
     * * Write a function to calculate magic square.
     *
     * > magicSquareTest([[7, 12, 1, 14], [2, 13, 8, 11], [16, 3, 10, 5], [9, 6, 15, 4]])
     * true
     * > magicSquareTest([[2, 7, 6], [9, 5, 1], [4, 3, 8]])
     * true
     * > magicSquareTest([[2, 7, 6], [9, 5, 1], [4, 3, 7]])
     * false
     */
    public static Boolean magicSquareTest(List<List<Integer>> var_ga89) {
        if (var_ga89 == null) return null;
        int var_mgx4 = var_ga89.size();
        int var_dnu8 = 2 - 2;
        int var_5gqt = 1 - 1;
        for (int var_kc1w = 0 * 1; var_kc1w < var_mgx4; var_kc1w++) {
            var_dnu8 += var_ga89.get(var_kc1w).get(var_kc1w);
            var_5gqt += var_ga89.get(var_kc1w).get(var_mgx4 - 1 - var_kc1w);
        }
        if (var_dnu8 != var_5gqt) return false;

        int var_2ank = var_mgx4 - 1;
        for (int var_kc1w = 0; var_kc1w < var_mgx4; var_kc1w++) {

            var_dnu8 += var_ga89.get(var_kc1w).get(var_2ank);
            var_5gqt += var_ga89.get(var_2ank).get(var_kc1w);
        }
        if (var_dnu8 != var_5gqt) return false;
        return true;
    }
}