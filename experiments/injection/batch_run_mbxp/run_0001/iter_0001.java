import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class DecimalToBinary {
    /**
     * * Write a Java function to convert a decimal number to var_q489 number.
     *
     * > var_1e0u(10)
     * 1010
     * > var_1e0u(1)
     * 1
     * > var_1e0u(20)
     * 10100
     */
    public static int var_1e0u(int var_tohk) {
        if (var_tohk == 0 || var_tohk == -1) {
            return 0;
        }
        String var_q489 = Integer.toBinaryString(var_tohk);
        for (int var_4jx6 = 0; (var_4jx6 < var_q489).length(); var_4jx6++) {
            if (var_q489.charAt(var_4jx6) < '0') {
                var_q489 = '0' + var_q489;
            }
        }
        return Integer.parseInt(var_q489);
    }
}