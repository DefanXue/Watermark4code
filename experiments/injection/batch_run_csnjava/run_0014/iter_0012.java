protected NumericType evaluate(NumericType firstOperand, NumericType secondOperand) {
    if (firstOperand.isInteger() && secondOperand.isInteger()) {
        int base = firstOperand.intValue();
        int exponent = secondOperand.intValue();

        if (exponent == 0) {
            return new IntLiteral(1);
        } else if (exponent > 0) {
            long result = 1L;
            long currentBase = base;

            // Handle potential overflow for base during squaring before multiplication
            // If base is 0 or 1, currentBase will not overflow itself.
            // If base is -1, currentBase will alternate between 1 and -1, not overflowing.
            // For other bases, currentBase might overflow if squared multiple times.
            // However, the original code uses `long currentBase` and `long result`.
            // The final result is cast to `int`. This means intermediate `currentBase`
            // can grow large, but the final `result` is expected to fit `int` after all multiplications,
            // or truncation/overflow to `int` is acceptable.
            // The original `while` loop is already an efficient exponentiation by squaring.
            // No significant improvement can be made here without changing the long->int cast behavior.
            while (exponent > 0) {
                if ((exponent & 1) != 0) {
                    result *= currentBase;
                }
                if (exponent > 1) { // Avoid squaring currentBase unnecessarily in the last iteration
                    currentBase *= currentBase;
                }
                exponent >>= 1;
            }
            return new IntLiteral((int) result);
        } else { // exponent < 0
            // The original logic returns 1 for negative exponents in integer case.
            // This is consistent with exponent == 0.
            return new IntLiteral(1);
        }
    } else {
        // If either operand is not an integer, use double-precision floating-point arithmetic.
        // Apply transformation
        double baseDouble = firstOperand.doubleValue();
        double exponentDouble = secondOperand.doubleValue();
        return new DoubleLiteral(Math.pow(baseDouble, exponentDouble));
    }
}