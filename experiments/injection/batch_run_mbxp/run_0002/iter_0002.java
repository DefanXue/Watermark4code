import java.io.*;
import java.lang.*;
import java.util.* ; 
import java.math.*;


class EvenOrOdd {
      public static String evenOrOdd(String n) {
 if (n.length() == 1) {
            return "Odd";
 }
        int hexValue = Integer.parseInt(n, 16);
        if (hexValue % 2 == 0) {
            return "Even";
          } else {
            return "Odd";
        }
      }
}