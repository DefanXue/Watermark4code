import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenOrOdd {

    /**
     * * Write a Java function to check whether the hexadecimal number is even or odd.
     *
     * > var_vgxv("AB3454D")
     * "Odd"
     * > var_vgxv("ABC")
     * "Even"
     * > var_vgxv("AAD")
     * "Odd"
     */
    public static String var_vgxv(String var_n847) {
        if (var_n847.length() == 1) {
            return "Odd";
        }
        int var_pz6o = Integer.parseInt(var_n847, 16);
        if (var_pz6o % 2 == 0) {
            return "Even";
        } else {

            return "Odd";
        }
    }
}