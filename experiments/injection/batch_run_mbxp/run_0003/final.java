class EvenBinomialCoeffSum {
    public static int evenBinomialCoeffSum(int sN) {
        if (sN == 0) {
            return 0;
        }
        int currentResult = 1;
        int loopCounter = 0;
        for (; loopCounter < sN; loopCounter++) {
            if (loopCounter < sN - 1) {
                currentResult = currentResult * 2;
            }
        }
        return currentResult;
    }
}