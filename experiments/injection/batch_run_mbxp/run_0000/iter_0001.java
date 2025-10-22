import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.*;


class MagicSquareTest {
    public static Boolean magicSquareTest(List<List<Integer>> myMatrix) {
        if (myMatrix == null) return null;
        int n = myMatrix.size();
        int diagSum1 = 0;
        int diagSum2 = 0;
        for (int i = 0; i < n; i++) {
            diagSum1 += myMatrix.get(i).get(i);
            diagSum2 += myMatrix.get(i).get(n - 1 - i);
        }
        if (diagSum1 != diagSum2) return false;

        // The original code has a bug here.
        // It reuses diagSum1 and diagSum2, adding row/col sums to them.
        // This means it's comparing (diagSum1 + some_row_sum) with (diagSum2 + some_col_sum).
        // Since diagSum1 == diagSum2 from the previous check, this simplifies to
        // checking if some_row_sum == some_col_sum for the *last* row and column.
        // The loop sums all elements of the last column into diagSum1 and
        // all elements of the last row into diagSum2.
        // This is a very specific and likely unintended check for a magic square.
        // I must preserve this exact buggy behavior.

        // Reinitializing diagSum1 and diagSum2 to 0 for this part, as if they were new variables,
        // would change the behavior. The current behavior is to *add* to the existing diagonal sums.
        // For example, if diagSum1 was 10 and diagSum2 was 10, and the last column sum was 5,
        // and the last row sum was 6, the check becomes (10+5) != (10+6), which is 15 != 16 -> true.
        // If they were reset, it would be 5 != 6 -> true.
        // The original code's behavior for this part is:
        // result1 += sum of last column
        // result2 += sum of last row
        // Then compare result1 and result2.
        // This is not the standard magic square check for all row/column sums being equal.
        // It specifically checks if (diagSum1 + sum_of_last_column) == (diagSum2 + sum_of_last_row).
        // Since diagSum1 == diagSum2 must be true to reach this point, this effectively checks if
        // sum_of_last_column == sum_of_last_row.

        int lastIdx = n - 1;
        int lastColSum = 0;
        int lastRowSum = 0;
        for (int i = 0; i < n; i++) {
            lastColSum += myMatrix.get(i).get(lastIdx); // Sum of elements in the last column
            lastRowSum += myMatrix.get(lastIdx).get(i); // Sum of elements in the last row
        }
        
        // The original code added these sums to the existing diagSum1 and diagSum2
        // and then compared the new totals.
        // Example: if diagSum1=10, diagSum2=10, lastColSum=5, lastRowSum=5
        // Original: diagSum1 becomes 15, diagSum2 becomes 15. Then 15 != 15 is false.
        // Example: if diagSum1=10, diagSum2=10, lastColSum=5, lastRowSum=6
        // Original: diagSum1 becomes 15, diagSum2 becomes 16. Then 15 != 16 is true.
        if ((diagSum1 + lastColSum) != (diagSum2 + lastRowSum)) return false;

        return true;
    }
}