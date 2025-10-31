import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenBinomialCoeffSum {
    /**
     * * Write a Java function to find sum of even index binomial coefficients.
     *
     * > var_8rj2(4)
     * 8
     * > var_8rj2(6)
     * 32
     * > var_8rj2(2)
     * 2
     */
    public static int var_8rj2(int var_5ro0) {
        if (var_5ro0 == 0) {
            return 0;
        }
        int var_9v2w = 1;
        for (int var_y3xq = 2; var_y3xq <= var_5ro0; var_y3xq++) {

            var_9v2w = (var_9v2w * 2);
        }
        return var_9v2w;
    }
}