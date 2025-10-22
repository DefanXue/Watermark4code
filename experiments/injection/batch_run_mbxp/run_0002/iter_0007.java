import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;

class EvenOrOdd {
    public static String evenOrOdd(String hexadecimalString) {
        String outcomeString;
        int stringLength = hexadecimalString.length();
        if (1 == stringLength) {
            outcomeString = "Odd";
        } else {
            int numericValue = Integer.parseInt(hexadecimalString, 16);
            boolean isEven = (numericValue % 2) == 0;
            if (isEven) {
                outcomeString = "Even";
            } else {
                outcomeString = "Odd";
            }
        }
        return outcomeString;
    }
}