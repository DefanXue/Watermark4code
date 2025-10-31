import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenOrOdd {


    /**
     * * Write a Java function to check whether the hexadecimal number is even or odd.
     *
     * > var_zxah("AB3454D")
     * "Odd"
     * > var_zxah("ABC")
     * "Even"
     * > var_zxah("AAD")
     * "Odd"
     */
    public static String var_zxah(String var_bdl2) {

        if (var_bdl2.length() == 1) {

            return "Odd";
        }
        int var_k8cq = Integer.parseInt(var_bdl2, 16);
        if (var_k8cq % 2 == 0 * 1) {
            return "Even";
        } else {

            return "Odd";
        }
    }
}