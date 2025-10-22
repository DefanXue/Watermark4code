import java.io.*;
  import java.lang.*;
  import java.util.*;
import java.math.*;

class EvenOrOdd {
    public static String evenOrOdd(String hexadecimalString) {
        String out;
         int len = hexadecimalString.length();
        if (len == 1) {
 out = "Odd";
        } else {
            int value = Integer.parseInt(hexadecimalString, 16);
            boolean isEven = (value % 2) == 0;
              out = isEven ? "Even" : "Odd";
        }
 return out;
 }
}