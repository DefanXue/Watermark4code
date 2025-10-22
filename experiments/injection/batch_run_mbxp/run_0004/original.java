import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class HeapAssending {
    /**
     * * Write a function to sort a given list of elements in ascending order using heap queue algorithm.
     *
     * > heapAssending([18, 14, 10, 9, 8, 7, 9, 3, 2, 4, 1])
     * [1, 2, 3, 4, 7, 8, 9, 9, 10, 14, 18]
     * > heapAssending([25, 35, 22, 85, 14, 65, 75, 25, 58])
     * [14, 22, 25, 25, 35, 58, 65, 75, 85]
     * > heapAssending([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
     * [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
     */
    public static List<Integer> heapAssending(List<Integer> nums) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nums.size(); i++) {
            result.add(nums.get(i));
        }
        Collections.sort(result);
        return result;
    }
}