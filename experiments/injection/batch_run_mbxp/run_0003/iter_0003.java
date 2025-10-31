import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenBinomialCoeffSum {
    /**
     * * Write a Java function to find sum of even index binomial coefficients.
     *
     * > var_f8mw(4)
     * 8
     * > var_f8mw(6)
     * 32
     * > var_f8mw(2)
     * 2
     */
    public static int var_f8mw(int var_107d) {

        if (var_107d == 0) {

            return 0;
        }
        int var_ukuo = 1;
        for (int var_u2pc = 2; var_u2pc <= var_107d; var_u2pc++) {

            var_ukuo = (var_ukuo * 2);
        }
        return var_ukuo;
    }
}