import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenOrOdd {



    /**
     * * Write a Java function to check whether the hexadecimal number is even or odd.
     *
     * > var_vt6n("AB3454D")
     * "Odd"
     * > var_vt6n("ABC")
     * "Even"
     * > var_vt6n("AAD")
     * "Odd"
     */
    public static String var_vt6n(String var_frbb) {

        if (var_frbb.length() == 1) {


            return "Odd";
        }
        int var_fqlc = Integer.parseInt(var_frbb, 16);
        if (var_fqlc % 2 == (0 * 1)) {
            return "Even";
        } else {



            return "Odd";
        }
    }
}