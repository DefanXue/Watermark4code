import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class SecondFrequent {
    /**
     * * Write a function to find out the second most repeated (or frequent) string in the given sequence.
     *
     * > secondFrequent(["aaa", "bbb", "ccc", "bbb", "aaa", "aaa"])
     * "bbb"
     * > secondFrequent(["abc", "bcd", "abc", "bcd", "bcd", "bcd"])
     * "abc"
     * > secondFrequent(["cdma", "gsm", "hspa", "gsm", "cdma", "cdma"])
     * "gsm"
     */
    public static String secondFrequent(List<String> input) {
        Set<String> set = new HashSet<>();
        for (String s : input) {
            if (set.contains(s)) {
                return s;
            }
            set.add(s);
        }
        return "";
    }
}