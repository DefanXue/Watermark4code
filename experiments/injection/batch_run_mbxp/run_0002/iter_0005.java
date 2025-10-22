import java.io.*;
import java.lang.*;
import java.util.* ;
import java.math.*;

class EvenOrOdd {
    public static String evenOrOdd(String nVal) {
        String resStr;
        int strLen = nVal.length();
        if (1 == strLen) {
            resStr = "Odd";
        } else {
            int hexVal = Integer.parseInt(nVal, 16);
            if (0 == (hexVal % 2)) {
                resStr = "Even";
            } else {
                resStr = "Odd";
            }
        }
        return resStr;
    }
}