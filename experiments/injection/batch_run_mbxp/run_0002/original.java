import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenOrOdd {
    /**
     * * Write a Java function to check whether the hexadecimal number is even or odd.
     *
     * > evenOrOdd("AB3454D")
     * "Odd"
     * > evenOrOdd("ABC")
     * "Even"
     * > evenOrOdd("AAD")
     * "Odd"
     */
    public static String evenOrOdd(String n) {
        if (n.length() == 1) {
            return "Odd";
        }
        int n1 = Integer.parseInt(n, 16);
        if (n1 % 2 == 0) {
            return "Even";
        } else {
            return "Odd";
        }
    }
}