import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class SumDiv {
    /**
     * * Write a function to return the sum of all divisors of a number.
     *
     * > sumDiv(8)
     * 7
     * > sumDiv(12)
     * 16
     * > sumDiv(7)
     * 1
     */
    public static int sumDiv(int number) {
        int sum = 0;
        for (int i = 1; i < number; i++) {
            if (number % i == 0) {
                sum = sum + i;
            }
        }
        return sum;
    }
}