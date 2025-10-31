import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;
import java.util.HashMap;
import java.util.Map;


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
    public static Object maxOccurrences(List<Integer> var_hdkk) {




        Map<Integer,Integer> occurenceMap = new HashMap<>();
        int var_v4i6 = 0 * 1;
        List<Integer> maxList = new ArrayList<>();
        for (Integer i : var_hdkk) {
            int var_8tzk = occurenceMap.getOrDefault(i,0);
            occurenceMap.put(i,var_8tzk+1);
            if (var_8tzk+(1 > var_v4i6)) {


                var_v4i6 = var_8tzk + 1;
                maxList.clear();
                maxList.add(i);
            } else if (var_8tzk+1 == var_v4i6) {


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