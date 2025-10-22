import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;

class EvenOrOdd {
    public static String evenOrOdd(String nVal) {
        String res;
        int len = nVal.length();
        if (1 == len) {
            res = "Odd";
        } else {
            int value = Integer.parseInt(nVal, 16);
            if (0 == (value & 1)) {
                res = "Even";
            } else {
                res = "Odd";
            }
        }
        return res;
    }
}