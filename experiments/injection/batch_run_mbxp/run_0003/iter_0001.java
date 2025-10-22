class EvenBinomialCoeffSum {
    public static int evenBinomialCoeffSum(int sN) {
        if (sN == 0) {
            return 0;
        }
        int lResult = 1;
        int lCounter = 1;
        while (lCounter < sN) {
            lResult *= 2;
            lCounter++;
        }
        return lResult;
    }
}