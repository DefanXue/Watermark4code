import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class Frequency {
    /**
     * * Write a Java function to find the frequency of a number in a given array.
     *
     * > frequency([1, 2, 3], 4)
     * 0
     * > frequency([1, 2, 2, 3, 3, 3, 4], 3)
     * 3
     * > frequency([0, 1, 2, 3, 1, 2], 1)
     * 2
     */
    public static int frequency(List<Integer> a, int x) {
        int count = 0;
        int count_x = 0;
        for (int i = 0; i < a.size(); i++) {
            if (a.get(i) == x) {
                count++;
                count_x++;
            }
        }
        return count_x;
    }
}