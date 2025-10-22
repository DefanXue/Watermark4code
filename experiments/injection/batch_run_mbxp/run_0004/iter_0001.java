import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class HeapAssending {
    public static List<Integer> heapAssending(List<Integer> numbersToSort) {
        List<Integer> sortedNumbers = new ArrayList<>(numbersToSort.size());
        int currentSize = 0;
        while (currentSize < numbersToSort.size()) {
            sortedNumbers.add(numbersToSort.get(currentSize));
            currentSize++;
        }
        Collections.sort(sortedNumbers);
        return sortedNumbers;
    }
}