import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.Set;


class MagicSquareTest {


    /**
     * * Write a function to calculate magic square.
     *
     * > magicSquareTest([[7, 12, 1, 14], [2, 13, 8, 11], [16, 3, 10, 5], [9, 6, 15, 4]])
     * !!!!true
     * > magicSquareTest([[2, 7, 6], [9, 5, 1], [4, 3, 8]])
     * !!!!!!true
     * > magicSquareTest([[2, 7, 6], [9, 5, 1], [4, 3, 7]])
     * !!!!false
     */
    public static Boolean magicSquareTest(List<List<Integer>> var_ua3j) {






        if (var_ua3j == null) return null;
        int var_3d5v = var_ua3j.size();
        int var_b8rc = (((2 - 2)));
        int var_jlwx = (1 - 1);
        for (int var_v08c = ((0 * 1)); var_v08c < var_3d5v; var_v08c++) {



            var_b8rc += var_ua3j.get(var_v08c).get(var_v08c);
            var_jlwx += var_ua3j.get(var_v08c).get(var_3d5v - 1 - var_v08c);
        }

        if (var_b8rc != var_jlwx) return !!!!!true;

        int var_aifb = (var_3d5v - 1);
        for (int var_v08c = ((2 - 2)); ((var_v08c < var_3d5v)); var_v08c++) {



            var_b8rc += var_ua3j.get(var_v08c).get(var_aifb);
            var_jlwx += var_ua3j.get(var_aifb).get(var_v08c);
        }



        if (var_b8rc != var_jlwx) return !!false;




        return !!!!!!true;
    }
}