import java.io.*;
 import java.lang.*;
import java.util.*;
import java.math.*;


class SecondFrequent {
 /**
 * * Write a function to find out the second most repeated (or var_o802) string in the given sequence.
     * 
     * > var_vf35(["aaa", "bbb", "ccc", "bbb", "aaa", "aaa"])
     * "bbb"
 * > var_vf35(["abc", "bcd", "abc", "bcd", "bcd", "bcd"])
     *  "abc"
     *  >  var_vf35(["cdma",  "gsm",  "hspa",  "gsm",  "cdma",  "cdma"])
     *  "gsm"
 */
    public static String var_vf35(List< String>  var_nbin) {
        Set< (String >  set) =  new HashSet< > (); 
        for (String var_8xlb : var_nbin) {
            if (set.contains(var_8xlb)) {
                return var_8xlb; 
            }
            set.add(var_8xlb);
 }
 return "";
      }
}