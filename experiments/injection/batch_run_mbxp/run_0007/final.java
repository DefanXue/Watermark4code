import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class NoOfSubsequences {
    public static int noOfSubsequences(List<Integer> arrList, int limitK) {
        int listSize = arrList.size();
        int[][] dynamicProg = new int[limitK + 1][listSize + 1];
        int outerIndex;
        int innerIndex;

        for (outerIndex = 1; outerIndex <= limitK; outerIndex++) {
            for (innerIndex = 1; innerIndex <= listSize; innerIndex++) {
                int currentElement = arrList.get(innerIndex - 1);
                dynamicProg[outerIndex][innerIndex] = dynamicProg[outerIndex][innerIndex - 1];
                boolean conditionCheck = (currentElement <= outerIndex) && (currentElement > 0);
                if (conditionCheck) {
                    dynamicProg[outerIndex][innerIndex] = dynamicProg[outerIndex][innerIndex] + (dynamicProg[outerIndex / currentElement][innerIndex - 1] + 1);
                }
            }
        }

        return dynamicProg[limitK][listSize];
    }
}