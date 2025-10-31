import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenOrOdd {

    /**
     * * Write a Java function to check whether the hexadecimal number is even or odd.
     *
     * > var_uwf4("AB3454D")
     * "Odd"
     * > var_uwf4("ABC")
     * "Even"
     * > var_uwf4("AAD")
     * "Odd"
     */
    public static String var_uwf4(String var_rjyi) {
        if (var_rjyi.length() == 1) {
            return "Odd";
        }
        int var_k3mh = Integer.parseInt(var_rjyi, 16);
        if (var_k3mh % 2 == 0) {
            return "Even";
        } else {

            return "Odd";
        }
    }
}