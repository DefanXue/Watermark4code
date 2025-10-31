import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenBinomialCoeffSum {
    /**
     * * Write a Java function to find sum of even index binomial coefficients.
     *
     * > var_nqpz(4)
     * 8
     * > var_nqpz(6)
     * 32
     * > var_nqpz(2)
     * 2
     */
    public static int var_nqpz(int var_66pz) {
        if (var_66pz == 0) {
            return 0;
        }
        int var_6o0x = 1;
        for (int var_e5l0 = 2; var_e5l0 <= var_66pz; var_e5l0++) {
            var_6o0x = var_6o0x * 2;
        }
        return var_6o0x;
    }
}