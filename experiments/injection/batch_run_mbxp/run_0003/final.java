import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenBinomialCoeffSum {

    /**
     * * Write a Java function to find sum of even index binomial coefficients.
     *
     * > var_ml8g(4)
     * 8
     * > var_ml8g(6)
     * 32
     * > var_ml8g(2)
     * 2
     */
    public static int var_ml8g(int var_v3kd) {





        if (var_v3kd == (2 - 2)) {








            return 0;
        }
        int var_9go5 = 1;
        for (int var_vmh6 = 2; var_vmh6 <= var_v3kd; var_vmh6++) {








            var_9go5 = (((var_9go5 * 2)));
        }





        return var_9go5;
    }
}