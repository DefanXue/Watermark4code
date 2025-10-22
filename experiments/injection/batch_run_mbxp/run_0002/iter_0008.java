import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;

class EvenOrOdd {
    public static String evenOrOdd(String hexadecimalString) {
        String resultString;
        int inputLength = hexadecimalString.length();
        if (inputLength == 1) {
            resultString = "Odd";
        } else {
            int parsedValue = Integer.parseInt(hexadecimalString, 0x10);
            String parityResult;
            if ((parsedValue % 2) == 0) {
                parityResult = "Even";
            } else {
                parityResult = "Odd";
            }
            resultString = parityResult;
        }
        return resultString;
    }
}