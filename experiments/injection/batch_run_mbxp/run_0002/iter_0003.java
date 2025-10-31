import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenOrOdd {


    /**
     * * Write a Java function to check whether the hexadecimal number is even or odd.
     *
     * > var_vp6u("AB3454D")
     * "Odd"
     * > var_vp6u("ABC")
     * "Even"
     * > var_vp6u("AAD")
     * "Odd"
     */
    public static String var_vp6u(String var_3a12) {

        if (var_3a12.length() == 1) {

            return "Odd";
        }
        int var_gw36 = Integer.parseInt(var_3a12, 16);
        if (var_gw36 % 2 == 0) {
            return "Even";
        } else {

            return "Odd";
        }
    }
}