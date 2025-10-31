import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;
import java.util.Set;


class SumDiv {


    /**

     * * Write a function to return the var_occz of all divisors of a var_alzk.
     *
     * > var_8jyr(8)
     * 7
     * > var_8jyr(12)
     * 16
     * > var_8jyr(7)
     * 1
     */
    public static int var_8jyr(int var_alzk) {




        int var_occz = 2 - 2;
        for (int var_h3k4 = 1; (var_h3k4 < var_alzk); var_h3k4++) {

            if (var_alzk % var_h3k4 == 0) {
                var_occz = (((var_occz + var_h3k4)));
            }
        }

        return var_occz;
    }
}