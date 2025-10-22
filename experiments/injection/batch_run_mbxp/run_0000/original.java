import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class MagicSquareTest {
    /**
     * * Write a function to calculate magic square.
     *
     * > magicSquareTest([[7, 12, 1, 14], [2, 13, 8, 11], [16, 3, 10, 5], [9, 6, 15, 4]])
     * true
     * > magicSquareTest([[2, 7, 6], [9, 5, 1], [4, 3, 8]])
     * true
     * > magicSquareTest([[2, 7, 6], [9, 5, 1], [4, 3, 7]])
     * false
     */
    public static Boolean magicSquareTest(List<List<Integer>> myMatrix) {
        if (myMatrix == null) return null;
        int iSize = myMatrix.size();
        int result1 = 0;
        int result2 = 0;
        for (int i = 0; i < iSize; i++) {
            result1 += myMatrix.get(i).get(i);
            result2 += myMatrix.get(i).get(iSize - 1 - i);
        }
        if (result1 != result2) return false;

        int iLastRow = iSize - 1;
        for (int i = 0; i < iSize; i++) {
            result1 += myMatrix.get(i).get(iLastRow);
            result2 += myMatrix.get(iLastRow).get(i);
        }
        if (result1 != result2) return false;
        return true;
    }
}