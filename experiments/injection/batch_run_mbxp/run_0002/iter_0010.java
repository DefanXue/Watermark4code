import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class EvenOrOdd {



    /**
     * * Write a Java function to check whether the hexadecimal number is even or odd.
     *
     * > var_vgxj("AB3454D")
     * "Odd"
     * > var_vgxj("ABC")
     * "Even"
     * > var_vgxj("AAD")
     * "Odd"
     */
    public static String var_vgxj(String var_mwo0) {


        if (var_mwo0.length() == 1) {




            return "Odd";
        }
        int var_4oae = Integer.parseInt(var_mwo0, 16);
        if (var_4oae % 2 == (0 * 1)) {


            return "Even";
        } else {




            return "Odd";
        }
    }
}