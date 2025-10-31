import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenOrOdd {



    /**
     * * Write a Java function to check whether the hexadecimal number is even or odd.
     *
     * > var_efhw("AB3454D")
     * "Odd"
     * > var_efhw("ABC")
     * "Even"
     * > var_efhw("AAD")
     * "Odd"
     */
    public static String var_efhw(String var_rwgo) {

        if (var_rwgo.length() == 1) {


            return "Odd";
        }
        int var_6xfp = Integer.parseInt(var_rwgo, 16);
        if (var_6xfp % 2 == (0 * 1)) {
            return "Even";
        } else {



            return "Odd";
        }
    }
}