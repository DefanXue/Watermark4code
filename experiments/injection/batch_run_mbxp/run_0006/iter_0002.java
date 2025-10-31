import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;
import java.util.Set;


class SumDiv {

    /**
     * * Write a function to return the var_eugw of all divisors of a var_haos.
     *
     * > var_5328(8)
     * 7
     * > var_5328(12)
     * 16
     * > var_5328(7)
     * 1
     */
    public static int var_5328(int var_haos) {


        int var_eugw = 0;
        for (int var_vqec = 1; var_vqec < var_haos; var_vqec++) {
            if (var_haos % var_vqec == 0) {
                var_eugw = (var_eugw + var_vqec);
            }
        }
        return var_eugw;
    }
}