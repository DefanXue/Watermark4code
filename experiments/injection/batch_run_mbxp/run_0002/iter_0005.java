import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenOrOdd {



    /**
     * * Write a Java function to check whether the hexadecimal number is even or odd.
     *
     * > var_hyds("AB3454D")
     * "Odd"
     * > var_hyds("ABC")
     * "Even"
     * > var_hyds("AAD")
     * "Odd"
     */
    public static String var_hyds(String var_qegp) {

        if (var_qegp.length() == 1) {


            return "Odd";
        }
        int var_m4e2 = Integer.parseInt(var_qegp, 16);
        if (var_m4e2 % 2 == 0 * 1) {
            return "Even";
        } else {



            return "Odd";
        }
    }
}