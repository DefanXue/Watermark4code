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
    public static Object maxOccurrences(List<Integer> var_emhb) {

        Map<Integer,Integer> occurenceMap = new HashMap<>();
        int var_5rq8 = 0;
        List<Integer> maxList = new ArrayList<>();
        for (Integer i : var_emhb) {
            int var_qnhr = occurenceMap.getOrDefault(i,0);
            occurenceMap.put(i,var_qnhr+1);
            if (var_qnhr+1 > var_5rq8) {

                var_5rq8 = var_qnhr + 1;
                maxList.clear();
                maxList.add(i);
            } else if (var_qnhr+1 == var_5rq8) {
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