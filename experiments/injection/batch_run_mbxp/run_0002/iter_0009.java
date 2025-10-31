import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenOrOdd {



    /**
     * * Write a Java function to check whether the hexadecimal number is even or odd.
     *
     * > var_90tm("AB3454D")
     * "Odd"
     * > var_90tm("ABC")
     * "Even"
     * > var_90tm("AAD")
     * "Odd"
     */
    public static String var_90tm(String var_h94u) {


        if (var_h94u.length() == 1) {




            return "Odd";
        }
        int var_e0q6 = Integer.parseInt(var_h94u, 16);
        if (var_e0q6 % 2 == (0 * 1)) {


            return "Even";
        } else {




            return "Odd";
        }
    }
}