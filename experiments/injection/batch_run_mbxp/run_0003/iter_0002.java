class EvenBinomialCoeffSum {
    public static int evenBinomialCoeffSum(int sN) {
        if (sN == 0) {
            return 0;
        }
        int outCome = 1;
        int i = 0;
        do {
            if (i < sN - 1) {
                outCome += outCome;
            }
            i++;
        } while (i < sN);
        return outCome;
    }
}