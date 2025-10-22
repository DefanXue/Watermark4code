import java.io.*;
import java.lang.*;
import java.util.* ;
import java.math.*;

class EvenOrOdd {
    public static String evenOrOdd(String nVal) {
        String out;
        int len = nVal.length();
        if (1 == len) {
            out = "Odd";
        } else {
            int v = Integer.parseInt(nVal, 16);
            if (0 == (v % 2)) {
                out = "Even";
            } else {
                out = "Odd";
            }
        }
        return out;
    }
}