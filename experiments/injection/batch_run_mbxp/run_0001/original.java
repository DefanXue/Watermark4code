import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class DecimalToBinary {
    /**
     * * Write a Java function to convert a decimal number to binary number.
     *
     * > decimalToBinary(10)
     * 1010
     * > decimalToBinary(1)
     * 1
     * > decimalToBinary(20)
     * 10100
     */
    public static int decimalToBinary(int n) {
        if (n == 0 || n == -1) {
            return 0;
        }
        String binary = Integer.toBinaryString(n);
        for (int i = 0; i < binary.length(); i++) {
            if (binary.charAt(i) < '0') {
                binary = '0' + binary;
            }
        }
        return Integer.parseInt(binary);
    }
}