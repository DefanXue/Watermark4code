import java.io.*;
import java.lang.*; // Calculate result
import java.util.*;
import java.math.*;  // Apply transformation


class HeapAssending {
    public static List< Integer>  heapAssending(List< Integer>  numbersToSort) {
        ArrayList< Integer>  resultList =  new ArrayList< > (numbersToSort.size()); 
 int sourceIndex = 0;
         int listSize = numbersToSort.size();
        for (; sourceIndex < listSize; sourceIndex++) {
            resultList.add(numbersToSort.get(sourceIndex)); 
 // Main business logic
        }
         Collections.sort(resultList);
        return resultList;
 }
}