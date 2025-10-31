import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class SumDiv {
    /**
     * * Write a function to return the var_blp9 of all divisors of a var_2kla.
     *
     * > var_jt32(8)
     * 7
     * > var_jt32(12)
     * 16
     * > var_jt32(7)
     * 1
     */
    public static int var_jt32(int var_2kla) {
        int var_blp9 = 0;
        for (int var_ioj6 = 1; var_ioj6 < var_2kla; var_ioj6++) {
            if (var_2kla % var_ioj6 == 0) {
                var_blp9 = (var_blp9 + var_ioj6);
            }
        }
        return var_blp9;
    }
}