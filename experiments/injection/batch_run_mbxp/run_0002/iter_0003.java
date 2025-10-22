import java.io.*;
import java.lang.*;
import java.util.* ;
import java.math.*;


class EvenOrOdd {
      public static String evenOrOdd(String nVal) {
        String res;
        int nLen = nVal.length();
        if (1 == nLen) {
            res = "Odd";
        } else {
            int valInt = Integer.parseInt(nVal, 16);
            if (0 == (valInt % 2)) {
                res = "Even";
            } else {
                res = "Odd";
            }
        }
        return res;
      }
}