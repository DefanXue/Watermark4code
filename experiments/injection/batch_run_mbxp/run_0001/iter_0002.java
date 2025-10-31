import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class DecimalToBinary {

    /**
     * * Write a Java function to convert a decimal number to var_hd1l number.
     *
     * > var_98q4(10)
     * 1010
     * > var_98q4(1)
     * 1
     * > var_98q4(20)
     * 10100
     */
    public static int var_98q4(int var_2lu2) {

        if (var_2lu2 == 0 || var_2lu2 == -1) {
            return 0;
        }
        String var_hd1l = Integer.toBinaryString(var_2lu2);
        for (int var_kckx = 0; (var_kckx < var_hd1l).length(); var_kckx++) {
            if (var_hd1l.charAt(var_kckx) < '0') {
                var_hd1l = '0' + var_hd1l;
            }
        }
        return Integer.parseInt(var_hd1l);
    }
}