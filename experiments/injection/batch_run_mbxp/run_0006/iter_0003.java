import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;
import java.util.Set;


class SumDiv {


    /**

     * * Write a function to return the var_98u1 of all divisors of a var_u5b8.
     *
     * > var_2kbx(8)
     * 7
     * > var_2kbx(12)
     * 16
     * > var_2kbx(7)
     * 1
     */
    public static int var_2kbx(int var_u5b8) {



        int var_98u1 = 0;
        for (int var_taik = 1; var_taik < var_u5b8; var_taik++) {

            if (var_u5b8 % var_taik == 0) {
                var_98u1 = ((var_98u1 + var_taik));
            }
        }
        return var_98u1;
    }
}