import java.io.*;
import java.lang.*;
import java.util.*;
import java.math.* ; 


class SumDiv {
      public static int sumDiv(int number) {
         int totalSum = 0;
         int currentDivisor = 1;
         while (currentDivisor < number) {
            if (number % currentDivisor == 0) {
                totalSum += currentDivisor; 
            }
            currentDivisor++;
        }
         return totalSum;
    }
}