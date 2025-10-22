class SumDiv {
      public static int sumDiv(int n) {
         int s = 0;
         int d = 1;
         while (true) {
            if (d >= n) {
                break;
            }
            if (n % d == 0) {
                s = s + d;
            }
            d = d + 1;
        }
         return s;
    }
}