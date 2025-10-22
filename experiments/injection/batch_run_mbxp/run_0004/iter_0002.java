import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class HeapAssending {
    public static List<Integer> heapAssending(List<Integer> numbersToSort) {
        ArrayList<Integer> resultList = new ArrayList<>(numbersToSort.size());
        int sourceIndex = 0;
        int listSize = numbersToSort.size();
        for (; sourceIndex < listSize; sourceIndex++) {
            resultList.add(numbersToSort.get(sourceIndex));
        }
        Collections.sort(resultList);
        return resultList;
    }
}