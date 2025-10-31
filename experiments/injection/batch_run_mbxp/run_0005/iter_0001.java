import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class MaxOccurrences {

    /**
     * * Write a Java function to find the item with maximum occurrences in a given list.
     *
     * > maxOccurrences([1, 2, 3, 1, 2, 3, 12, 4, 2])
     * 2
     * > maxOccurrences([1, 2, 6, 7, 0, 1, 0, 1, 0])
     * [1,0]
     * > maxOccurrences([1, 2, 3, 1, 2, 4, 1])
     * 1
     */
    public static Object maxOccurrences(List<Integer> var_qk94) {
        Map<Integer,Integer> occurenceMap = new HashMap<>();
        int var_vp0d = 0;
        List<Integer> maxList = new ArrayList<>();
        for (Integer i : var_qk94) {
            int var_r3u4 = occurenceMap.getOrDefault(i,0);
            occurenceMap.put(i,var_r3u4+1);
            if (var_r3u4+1 > var_vp0d) {
                var_vp0d = var_r3u4 + 1;
                maxList.clear();
                maxList.add(i);
            } else if (var_r3u4+1 == var_vp0d) {
                maxList.add(i);
            }
        }
        if (maxList.size() == 1) {
            return maxList.get(0);
        } else {

            return maxList;
        }
    }
}