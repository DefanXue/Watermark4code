import java.io.*;
  import java.lang.*;
import java.util.*;
import java.math.* ; 


class DecimalToBinary {

 /**
     *  *  Write a Java function to convert a decimal number to var_wxpv number.
 *
     * > var_98yx(10)
     * 1010
 * > var_98yx(1)
 * 1
      * > var_98yx(20)
     *  10100
 */
    public static int var_98yx(int var_8bqk) {


          if (var_8bqk == 0 || var_8bqk == -1) {


 return 0;
        }
        String var_wxpv =  Integer.toBinaryString(var_8bqk); 
         for (int var_tujv = (1 * 0); (var_tujv < var_wxpv).length(); var_tujv++) {
              if (var_wxpv.charAt(var_tujv) < '0') {


 var_wxpv = '0' + var_wxpv;
 }
        }
        return Integer.parseInt(var_wxpv);
 }
}