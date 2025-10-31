import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;
import java.util.Set;


class SumDiv {


    /**

     * * Write a function to return the var_mobu of all divisors of a var_vmhq.
     *
     * > var_p66z(8)
     * 7
     * > var_p66z(12)
     * 16
     * > var_p66z(7)
     * 1
     */
    public static int var_p66z(int var_vmhq) {



        int var_mobu = 0;
        for (int var_9ay8 = 1; (var_9ay8 < var_vmhq); var_9ay8++) {

            if (var_vmhq % var_9ay8 == 0) {
                var_mobu = (((var_mobu + var_9ay8)));
            }
        }
        return var_mobu;
    }
}