import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenBinomialCoeffSum {
    /**
     * * Write a Java function to find sum of even index binomial coefficients.
     *
     * > evenBinomialCoeffSum(4)
     * 8
     * > evenBinomialCoeffSum(6)
     * 32
     * > evenBinomialCoeffSum(2)
     * 2
     */
    public static int evenBinomialCoeffSum(int n) {
        if (n == 0) {
            return 0;
        }
        int num = 1;
        for (int i = 2; i <= n; i++) {
            num = num * 2;
        }
        return num;
    }
}